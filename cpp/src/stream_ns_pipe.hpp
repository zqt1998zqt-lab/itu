#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace itu {

static_assert(sizeof(float) == sizeof(std::uint32_t),
              "external NS stream requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559,
              "external NS stream requires IEEE-754 float");
static_assert(std::endian::native == std::endian::little,
              "external NS stream requires a little-endian host");

// Parent-owned POSIX pipe transport for one stateful external NS frame.
// `worker_input_fd` is written by C++; `worker_output_fd` is read from the
// worker. The process owns neither descriptor and closes neither descriptor.
class StreamNsPipe {
 public:
  StreamNsPipe(int worker_input_fd, int worker_output_fd,
               unsigned frame_samples = 160)
      : worker_input_fd_(worker_input_fd),
        worker_output_fd_(worker_output_fd),
        frame_samples_(frame_samples),
        wire_(static_cast<std::size_t>(frame_samples) * sizeof(float)) {
#if defined(__unix__) || defined(__APPLE__)
    if (frame_samples_ == 0 || frame_samples_ > 4096)
      throw std::invalid_argument(
          "NS stream frame size must be in 1..4096 samples");
    if (worker_input_fd_ < 3 || worker_output_fd_ < 3 ||
        worker_input_fd_ == worker_output_fd_)
      throw std::invalid_argument(
          "NS stream fds must be distinct real descriptors >= 3");
    const int input_flags = ::fcntl(worker_input_fd_, F_GETFL);
    const int output_flags = ::fcntl(worker_output_fd_, F_GETFL);
    if (input_flags < 0 || output_flags < 0)
      throw std::invalid_argument("NS stream fd validation failed");
    struct stat input_status {};
    struct stat output_status {};
    if (::fstat(worker_input_fd_, &input_status) != 0 ||
        ::fstat(worker_output_fd_, &output_status) != 0 ||
        !S_ISFIFO(input_status.st_mode) || !S_ISFIFO(output_status.st_mode))
      throw std::invalid_argument("NS stream fds must be POSIX pipes");
    const int input_mode = input_flags & O_ACCMODE;
    const int output_mode = output_flags & O_ACCMODE;
    if ((input_mode != O_WRONLY && input_mode != O_RDWR) ||
        (output_mode != O_RDONLY && output_mode != O_RDWR))
      throw std::invalid_argument(
          "NS stream input fd must be writable and output fd readable");

    struct sigaction ignored {};
    ignored.sa_handler = SIG_IGN;
    if (::sigemptyset(&ignored.sa_mask) != 0 ||
        ::sigaction(SIGPIPE, &ignored, &previous_sigpipe_) != 0)
      throw std::runtime_error("cannot install scoped SIGPIPE handling");
    sigpipe_scoped_ = true;
#else
    (void)worker_input_fd_;
    (void)worker_output_fd_;
    throw std::runtime_error(
        "external NS stream pipes are supported only on POSIX");
#endif
  }

  ~StreamNsPipe() {
#if defined(__unix__) || defined(__APPLE__)
    if (sigpipe_scoped_) ::sigaction(SIGPIPE, &previous_sigpipe_, nullptr);
#endif
  }

  StreamNsPipe(const StreamNsPipe&) = delete;
  StreamNsPipe& operator=(const StreamNsPipe&) = delete;

  bool process(const float* input, float* output, unsigned samples) noexcept {
    if (!input || !output || samples != frame_samples_) return false;
    for (unsigned i = 0; i < samples; ++i)
      if (!std::isfinite(input[i])) return false;
    std::memcpy(wire_.data(), input, wire_.size());
#if defined(__unix__) || defined(__APPLE__)
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    if (!write_all(wire_.data(), wire_.size(), deadline)) return false;
    if (!read_all(wire_.data(), wire_.size(), deadline)) return false;
#else
    return false;
#endif
    std::memcpy(output, wire_.data(), wire_.size());
    for (unsigned i = 0; i < samples; ++i)
      if (!std::isfinite(output[i])) return false;
    return true;
  }

 private:
#if defined(__unix__) || defined(__APPLE__)
  static int poll_until(int fd, short events,
                        std::chrono::steady_clock::time_point deadline) {
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) return 0;
      const auto remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
      const auto bounded = std::min<std::int64_t>(
          remaining.count(), std::numeric_limits<int>::max());
      struct pollfd descriptor {fd, events, 0};
      const int status =
          ::poll(&descriptor, 1, static_cast<int>(std::max<std::int64_t>(1, bounded)));
      if (status < 0) {
        if (errno == EINTR) continue;
        return -1;
      }
      if (status == 0) return 0;
      if (descriptor.revents & (POLLERR | POLLNVAL)) return -1;
      if (events & POLLIN) {
        if (descriptor.revents & POLLIN) return 1;
        if (descriptor.revents & POLLHUP) return 0;
      } else {
        if (descriptor.revents & POLLOUT) return 1;
        if (descriptor.revents & POLLHUP) return -1;
      }
    }
  }

  bool write_all(const unsigned char* data, std::size_t size,
                 std::chrono::steady_clock::time_point deadline) noexcept {
    std::size_t offset = 0;
    while (offset < size) {
      const int ready = poll_until(worker_input_fd_, POLLOUT, deadline);
      if (ready != 1) return false;
      const ssize_t written =
          ::write(worker_input_fd_, data + offset, size - offset);
      if (written < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        return false;
      }
      if (written == 0) return false;
      offset += static_cast<std::size_t>(written);
    }
    return true;
  }

  bool read_all(unsigned char* data, std::size_t size,
                std::chrono::steady_clock::time_point deadline) noexcept {
    std::size_t offset = 0;
    while (offset < size) {
      const int ready = poll_until(worker_output_fd_, POLLIN, deadline);
      if (ready != 1) return false;
      const ssize_t received =
          ::read(worker_output_fd_, data + offset, size - offset);
      if (received < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        return false;
      }
      if (received == 0) return false;
      offset += static_cast<std::size_t>(received);
    }
    return true;
  }
#endif

  int worker_input_fd_ = -1;
  int worker_output_fd_ = -1;
  unsigned frame_samples_ = 0;
  std::vector<unsigned char> wire_;
#if defined(__unix__) || defined(__APPLE__)
  struct sigaction previous_sigpipe_ {};
  bool sigpipe_scoped_ = false;
#endif
};

}  // namespace itu
