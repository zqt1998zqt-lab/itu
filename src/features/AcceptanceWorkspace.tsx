import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { LoaderCircle } from "lucide-react";
import { PageBoundary } from "../components/PageBoundary";
import type { FiveSignalMode } from "./five-signal-flow-model";

const ItuAlignment = lazy(() => import("./ItuAlignment"));
const BasicUplinkTest = lazy(() => import("./BasicUplinkTest"));
const BasicDownlinkTest = lazy(() => import("./BasicDownlinkTest"));
const EchoTest = lazy(() => import("./EchoTest"));
const DoubleTalkTest = lazy(() => import("./DoubleTalkTest"));
const BackgroundNoiseTest = lazy(() => import("./BackgroundNoiseTest"));
const BackgroundNoiseIdleTest = lazy(() => import("./BackgroundNoiseIdleTest"));
const BackgroundNoiseUrbanTest = lazy(() => import("./BackgroundNoiseUrbanTest"));
const BackgroundNoiseHighwayTest = lazy(() => import("./BackgroundNoiseHighwayTest"));
const BackgroundNoiseHvacTest = lazy(() => import("./BackgroundNoiseHvacTest"));

export const ACCEPTANCE_VIEWS = [
  "alignment",
  "basic-uplink",
  "basic-downlink",
  "echo",
  "double-talk",
  "background-noise",
  "background-noise-idle",
  "background-noise-urban",
  "background-noise-highway",
  "background-noise-hvac",
] as const;

export type AcceptanceView = (typeof ACCEPTANCE_VIEWS)[number];

const ACCEPTANCE_VIEW_SET: ReadonlySet<string> = new Set(ACCEPTANCE_VIEWS);

export function isAcceptanceView(value: string): value is AcceptanceView {
  return ACCEPTANCE_VIEW_SET.has(value);
}

type BusyByView = Record<AcceptanceView, boolean>;
type BusySetter = (busy: boolean) => void;

export default function AcceptanceWorkspace({
  activeView,
}: {
  activeView: AcceptanceView | null;
}) {
  const [openedViews, setOpenedViews] = useState<AcceptanceView[]>([]);
  const [algorithmClusterId, setAlgorithmClusterId] = useState("vcp_unified");
  const [mode, setMode] = useState<FiveSignalMode>("WB-wired");
  const [busyByView, setBusyByView] = useState<BusyByView>({
    alignment: false,
    "basic-uplink": false,
    "basic-downlink": false,
    echo: false,
    "double-talk": false,
    "background-noise": false,
    "background-noise-idle": false,
    "background-noise-urban": false,
    "background-noise-highway": false,
    "background-noise-hvac": false,
  });

  useEffect(() => {
    if (activeView) {
      setOpenedViews((current) => current.includes(activeView) ? current : [...current, activeView]);
    }
  }, [activeView]);

  const busySetters = useMemo(() => {
    const setterFor = (view: AcceptanceView): BusySetter => (busy) => {
      setBusyByView((current) => current[view] === busy ? current : { ...current, [view]: busy });
    };
    return {
      alignment: setterFor("alignment"),
      "basic-uplink": setterFor("basic-uplink"),
      "basic-downlink": setterFor("basic-downlink"),
      echo: setterFor("echo"),
      "double-talk": setterFor("double-talk"),
      "background-noise": setterFor("background-noise"),
      "background-noise-idle": setterFor("background-noise-idle"),
      "background-noise-urban": setterFor("background-noise-urban"),
      "background-noise-highway": setterFor("background-noise-highway"),
      "background-noise-hvac": setterFor("background-noise-hvac"),
    } satisfies Record<AcceptanceView, BusySetter>;
  }, []);

  const externalRunning = (view: AcceptanceView) =>
    ACCEPTANCE_VIEWS.some((candidate) => candidate !== view && busyByView[candidate]);

  const loading = <p className="page-loading" role="status"><LoaderCircle size={18} className="spin" />正在加载页面…</p>;

  // Keep each visited page mounted so navigation preserves task ownership and polling.
  return <>
    {activeView === "alignment" || openedViews.includes("alignment") ? (
      <div hidden={activeView !== "alignment"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <ItuAlignment
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              selectedMode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("alignment")}
              onBusyChange={busySetters.alignment}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "basic-uplink" || openedViews.includes("basic-uplink") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "basic-uplink"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BasicUplinkTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("basic-uplink")}
              onBusyChange={busySetters["basic-uplink"]}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "basic-downlink" || openedViews.includes("basic-downlink") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "basic-downlink"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BasicDownlinkTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("basic-downlink")}
              onBusyChange={busySetters["basic-downlink"]}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "echo" || openedViews.includes("echo") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "echo"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <EchoTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("echo")}
              onBusyChange={busySetters.echo}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "double-talk" || openedViews.includes("double-talk") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "double-talk"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <DoubleTalkTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("double-talk")}
              onBusyChange={busySetters["double-talk"]}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "background-noise" || openedViews.includes("background-noise") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "background-noise"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BackgroundNoiseTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("background-noise")}
              onBusyChange={busySetters["background-noise"]}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "background-noise-idle" || openedViews.includes("background-noise-idle") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "background-noise-idle"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BackgroundNoiseIdleTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("background-noise-idle")}
              onBusyChange={busySetters["background-noise-idle"]}
              pageVisible={activeView === "background-noise-idle"}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "background-noise-urban" || openedViews.includes("background-noise-urban") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "background-noise-urban"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BackgroundNoiseUrbanTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("background-noise-urban")}
              onBusyChange={busySetters["background-noise-urban"]}
              pageVisible={activeView === "background-noise-urban"}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "background-noise-highway" || openedViews.includes("background-noise-highway") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "background-noise-highway"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BackgroundNoiseHighwayTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("background-noise-highway")}
              onBusyChange={busySetters["background-noise-highway"]}
              pageVisible={activeView === "background-noise-highway"}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
    {activeView === "background-noise-hvac" || openedViews.includes("background-noise-hvac") ? (
      <div className="evaluation-page basic-uplink-page" hidden={activeView !== "background-noise-hvac"}>
        <PageBoundary>
          <Suspense fallback={loading}>
            <BackgroundNoiseHvacTest
              algorithmClusterId={algorithmClusterId}
              onAlgorithmClusterChange={setAlgorithmClusterId}
              mode={mode}
              onModeChange={setMode}
              externalRunning={externalRunning("background-noise-hvac")}
              onBusyChange={busySetters["background-noise-hvac"]}
              pageVisible={activeView === "background-noise-hvac"}
            />
          </Suspense>
        </PageBoundary>
      </div>
    ) : null}
  </>;
}
