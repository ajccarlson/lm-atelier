import { useCallback, useState } from "react";

import type { RoutingMode } from "./types";

export function mediaOutputCountForTurn(
  mode: RoutingMode,
  outputCount: number,
): number | undefined {
  return (mode === "image" || mode === "video") && outputCount > 1
    ? outputCount
    : undefined;
}

export function useMediaOutputCount() {
  const [outputCount, setOutputCount] = useState(1);
  const resetOutputCount = useCallback(() => setOutputCount(1), []);
  return { outputCount, setOutputCount, resetOutputCount };
}
