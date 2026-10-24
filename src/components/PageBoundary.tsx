import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

export class PageBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("页面加载失败", error, info.componentStack);
  }

  render() {
    if (this.state.failed) return <section className="panel" role="alert">
      <p>页面加载失败。请刷新重试，也可通过导航查看其他页面。</p>
      <button className="button secondary" onClick={() => window.location.reload()}>刷新页面</button>
    </section>;
    return this.props.children;
  }
}
