import { useReducer, useCallback, useRef, useEffect } from 'react';

const MAX_HISTORY = 50;

function reducer(state, action) {
  switch (action.type) {
    case 'PUSH': {
      const past = [...state.past, state.present].slice(-MAX_HISTORY);
      return { past, present: action.snapshot, future: [] };
    }
    case 'UNDO': {
      if (!state.past.length) return state;
      return {
        past: state.past.slice(0, -1),
        present: state.past[state.past.length - 1],
        future: [state.present, ...state.future],
      };
    }
    case 'REDO': {
      if (!state.future.length) return state;
      return {
        past: [...state.past, state.present],
        present: state.future[0],
        future: state.future.slice(1),
      };
    }
    case 'RESET':
      return { past: [], present: action.snapshot, future: [] };
    default:
      return state;
  }
}

/**
 * Undo/redo hook for arbitrary snapshot objects.
 *
 * @param {*} initialSnapshot  Initial state to seed history with.
 * @returns {{
 *   snapshot: *,
 *   canUndo: boolean,
 *   canRedo: boolean,
 *   pushSnapshot: (s: *) => void,
 *   undo: () => void,
 *   redo: () => void,
 *   reset: (s: *) => void,
 *   isProgrammaticRef: React.MutableRefObject<boolean>,
 * }}
 */
export function useUndoRedo(initialSnapshot) {
  const [{ past, present, future }, dispatch] = useReducer(reducer, {
    past: [],
    present: initialSnapshot,
    future: [],
  });

  // Callers set this to true before triggering a state change that comes
  // from undo/redo — the push-snapshot effect checks this flag and skips.
  const isProgrammaticRef = useRef(false);

  const pushSnapshot = useCallback((snapshot) => {
    dispatch({ type: 'PUSH', snapshot });
  }, []);

  const undo = useCallback(() => {
    isProgrammaticRef.current = true;
    dispatch({ type: 'UNDO' });
  }, []);

  const redo = useCallback(() => {
    isProgrammaticRef.current = true;
    dispatch({ type: 'REDO' });
  }, []);

  const reset = useCallback((snapshot) => {
    dispatch({ type: 'RESET', snapshot });
  }, []);

  // Bind Ctrl+Z / Ctrl+Y / Ctrl+Shift+Z globally.
  // Ignored when focus is inside a text-editing element.
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      const editable = document.activeElement?.isContentEditable;
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || editable) return;

      if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
        e.preventDefault();
        isProgrammaticRef.current = true;
        dispatch({ type: 'UNDO' });
      } else if (
        (e.key === 'y' && (e.ctrlKey || e.metaKey)) ||
        (e.key === 'z' && (e.ctrlKey || e.metaKey) && e.shiftKey)
      ) {
        e.preventDefault();
        isProgrammaticRef.current = true;
        dispatch({ type: 'REDO' });
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  return {
    snapshot: present,
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    pushSnapshot,
    undo,
    redo,
    reset,
    isProgrammaticRef,
  };
}
