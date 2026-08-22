import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { mediaOutputCountForTurn, useMediaOutputCount } from "./mediaOutputCount";
import { OutputCountControl } from "./OutputCountControl";

afterEach(cleanup);

describe("OutputCountControl", () => {
  it("uses the server cap and reports the selected count accessibly", () => {
    const onChange = vi.fn();
    render(
      <OutputCountControl mode="video" maximum={4} value={1} onChange={onChange} />,
    );

    const control = screen.getByRole("combobox", { name: "Number of outputs" });
    expect(Array.from((control as HTMLSelectElement).options, (option) => option.value))
      .toEqual(["1", "2", "3", "4"]);
    fireEvent.change(control, { target: { value: "3" } });
    expect(onChange).toHaveBeenCalledWith(3);
  });

  it("is absent for Auto and text", () => {
    const { rerender } = render(
      <OutputCountControl mode="auto" maximum={4} value={4} onChange={vi.fn()} />,
    );
    expect(screen.queryByRole("combobox", { name: "Number of outputs" }))
      .not.toBeInTheDocument();

    rerender(<OutputCountControl mode="text" maximum={4} value={4} onChange={vi.fn()} />);
    expect(screen.queryByRole("combobox", { name: "Number of outputs" }))
      .not.toBeInTheDocument();
  });

  it("keeps defensive rendering inside the server schema bounds", () => {
    const { rerender } = render(
      <OutputCountControl mode="image" maximum={0} value={1} onChange={vi.fn()} />,
    );
    expect(screen.getByRole("combobox", { name: "Number of outputs" }))
      .toHaveLength(1);

    rerender(<OutputCountControl mode="image" maximum={100} value={1} onChange={vi.fn()} />);
    expect(screen.getByRole("combobox", { name: "Number of outputs" }))
      .toHaveLength(16);
  });
});

describe("media output-count state", () => {
  it("defaults to one and resets after a turn", () => {
    const { result } = renderHook(useMediaOutputCount);
    expect(result.current.outputCount).toBe(1);

    act(() => result.current.setOutputCount(4));
    expect(result.current.outputCount).toBe(4);
    act(() => result.current.resetOutputCount());
    expect(result.current.outputCount).toBe(1);
  });

  it("emits only deliberate multi-output media requests", () => {
    expect(mediaOutputCountForTurn("image", 4)).toBe(4);
    expect(mediaOutputCountForTurn("video", 2)).toBe(2);
    expect(mediaOutputCountForTurn("image", 1)).toBeUndefined();
    expect(mediaOutputCountForTurn("auto", 4)).toBeUndefined();
    expect(mediaOutputCountForTurn("text", 4)).toBeUndefined();
  });
});
