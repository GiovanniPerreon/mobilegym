import type { NavigationDeclaration } from './navigation.types';

export const NAVIGATION_DECLARATION = {
  app: 'health_monitor',
  routes: [
    {
      path: '/',
      component: 'HomePage',
      entryPoint: 'home',
      uiStates: [{ id: 'health.home', description: 'Main dashboard' }],
    },
    {
      path: '/details',
      component: 'HealthDetailPage',
      entryPoint: 'none',
      uiStates: [{ id: 'health.details', description: 'Detailed report' }],
    },
  ],
  transitions: [
    {
      id: 'open_details',
      from: '/',
      to: '/details',
      mode: 'push',
    },
    {
      id: 'go_back',
      from: '/details',
      to: '/',
      mode: 'pop',
    },
  ],
  capabilities: { historyBack: true },
} as const satisfies NavigationDeclaration;

export type TransitionId = typeof NAVIGATION_DECLARATION.transitions[number]['id'];