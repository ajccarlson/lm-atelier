import { ChevronDown } from "lucide-react";

import type { RoutingMode } from "./types";

const MAX_OUTPUTS = 16;

export function OutputCountControl({
  mode,
  maximum,
  value,
  onChange,
}: {
  mode: RoutingMode;
  maximum: number;
  value: number;
  onChange: (value: number) => void;
}) {
  if (mode !== "image" && mode !== "video") return null;
  const limit = Math.min(MAX_OUTPUTS, Math.max(1, Math.floor(maximum)));
  return (
    <label className="mode-select output-count-select">
      <span>Outputs</span>
      <select
        aria-label="Number of outputs"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {Array.from(
          { length: limit },
          (_item, index) => index + 1,
        ).map((count) => <option key={count} value={count}>{count}</option>)}
      </select>
      <ChevronDown size={13} />
    </label>
  );
}
