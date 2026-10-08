import { useCallback, useEffect, useRef, useState } from "react";

// Fetch with loading / error / retry and request cancellation on unmount or dependency change.
export function useApi(fn, deps = []) {
  const [state, setState] = useState({ data: null, loading: true, error: null });
  const [tick, setTick] = useState(0);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    const ctrl = new AbortController();
    setState((s) => ({ ...s, loading: true, error: null }));
    fnRef
      .current({ signal: ctrl.signal })
      .then((data) => setState({ data, loading: false, error: null }))
      .catch((e) => {
        if (e.name !== "AbortError") setState({ data: null, loading: false, error: e });
      });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);

  const retry = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, retry };
}
