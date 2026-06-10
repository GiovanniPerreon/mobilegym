import type { NAVIGATION_DECLARATION } from './navigation.declaration';

export type NavigationDeclaration = typeof NAVIGATION_DECLARATION;
export type TransitionId = NavigationDeclaration['transitions'][number]['id'];
export type RoutePath = NavigationDeclaration['routes'][number]['path'];