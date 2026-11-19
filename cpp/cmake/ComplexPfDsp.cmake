# The HiFi adapter must implement itu/complex_pf_dsp_provider.h. Host builds
# remain usable without an SDK; selecting the DSP route then fails explicitly.
set(ITU_COMPLEX_PF_DSP_LIBRARY "" CACHE FILEPATH "HiFi Complex_PF provider or host-to-HiFi transport library")
if(ITU_COMPLEX_PF_DSP_LIBRARY)
  if(NOT EXISTS "${ITU_COMPLEX_PF_DSP_LIBRARY}")
    message(FATAL_ERROR "ITU_COMPLEX_PF_DSP_LIBRARY does not exist")
  endif()
  target_link_libraries(itu_audio PRIVATE "${ITU_COMPLEX_PF_DSP_LIBRARY}")
  target_compile_definitions(itu_audio PRIVATE ITU_HAS_COMPLEX_PF_DSP=1)
endif()
