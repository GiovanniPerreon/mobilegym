// apps/HealthMonitor/navigation.ts
import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { NAVIGATION_DECLARATION, type TransitionId } from './navigation.declaration';

export function useAppNavigate() {
  const navigate = useNavigate();

  const go = useCallback(
    (
      id: TransitionId,
      params: Record<string, string | number> = {},
      options?: { mode?: 'push' | 'replace' }
    ) => {
      const transition = NAVIGATION_DECLARATION.transitions.find(t => t.id === id);
      if (!transition) {
        throw new Error(`Transition not found: ${id}`);
      }

      // Handle pop mode by going back in history
      if (transition.mode === 'pop') {
        navigate(-1);
        return;
      }

      let targetPath = transition.to;
      Object.entries(params).forEach(([key, value]) => {
        targetPath = targetPath.replace(`:${key}`, String(value));
      });

      const effectiveMode = options?.mode ?? transition.mode;
      if (effectiveMode === 'replace') {
        navigate(targetPath, { replace: true });
      } else {
        navigate(targetPath);
      }
    },
    [navigate]
  );

  const back = useCallback((steps: number = 1) => {
    navigate(-steps);
  }, [navigate]);

  return { go, back };
}