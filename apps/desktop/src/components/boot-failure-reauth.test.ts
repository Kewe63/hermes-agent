import { describe, expect, it } from 'vitest'

import type { DesktopConnectionConfig } from '@/global'

import { deriveProviderShape, isRemoteReauthFailure, signInLabel } from './boot-failure-reauth'

function config(overrides: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    envOverride: false,
    mode: 'remote',
    profile: null,
    remoteAuthMode: 'oauth',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: 'https://box:9119',
    ...overrides
  }
}

describe('isRemoteReauthFailure', () => {
  it('true for a remote, gated, disconnected gateway with a URL', () => {
    expect(isRemoteReauthFailure(config())).toBe(true)
  })

  it('false when the oauth session is still connected', () => {
    expect(isRemoteReauthFailure(config({ remoteOauthConnected: true }))).toBe(false)
  })

  it('false for a local gateway', () => {
    expect(isRemoteReauthFailure(config({ mode: 'local' }))).toBe(false)
  })

  it('false for a token (non-gated) remote gateway', () => {
    expect(isRemoteReauthFailure(config({ remoteAuthMode: 'token' }))).toBe(false)
  })

  it('true for a basic-auth remote gateway with a URL (root cause for #51873)', () => {
    // 'basic' is not yet in DesktopConnectionConfig['remoteAuthMode'] — this
    // test pins the detection behavior we want once a follow-up PR widens the
    // type. The double cast (object → unknown → typed config) keeps TS happy
    // without committing to the type widening in the same PR.
    const basicConfig = config({
      remoteAuthMode: 'basic' as unknown as DesktopConnectionConfig['remoteAuthMode']
    })
    expect(isRemoteReauthFailure(basicConfig)).toBe(true)
  })

  it('false when a basic-auth remote has no URL to sign in against', () => {
    const basicConfig = config({
      remoteAuthMode: 'basic' as unknown as DesktopConnectionConfig['remoteAuthMode'],
      remoteUrl: ''
    })
    expect(isRemoteReauthFailure(basicConfig)).toBe(false)
  })

  it('false when there is no remote URL to sign in against', () => {
    expect(isRemoteReauthFailure(config({ remoteUrl: '' }))).toBe(false)
  })

  it('false for null/undefined config', () => {
    expect(isRemoteReauthFailure(null)).toBe(false)
    expect(isRemoteReauthFailure(undefined)).toBe(false)
  })
})

describe('deriveProviderShape', () => {
  it('generic copy when there are no providers', () => {
    expect(deriveProviderShape([])).toEqual({ isPassword: false, providerLabel: 'your identity provider' })
    expect(deriveProviderShape(null)).toEqual({ isPassword: false, providerLabel: 'your identity provider' })
  })

  it('password shape when the sole provider supports password', () => {
    expect(
      deriveProviderShape([{ name: 'basic', displayName: 'Username & Password', supportsPassword: true }])
    ).toEqual({ isPassword: true, providerLabel: 'Username & Password' })
  })

  it('OAuth shape when the provider is a redirect IDP', () => {
    expect(deriveProviderShape([{ name: 'nous', displayName: 'Nous Research', supportsPassword: false }])).toEqual({
      isPassword: false,
      providerLabel: 'Nous Research'
    })
  })

  it('mixed deployment keeps generic OAuth copy (not every provider is password)', () => {
    const shape = deriveProviderShape([
      { name: 'basic', displayName: 'Username & Password', supportsPassword: true },
      { name: 'nous', displayName: 'Nous Research', supportsPassword: false }
    ])

    expect(shape.isPassword).toBe(false)
    expect(shape.providerLabel).toBe('Username & Password / Nous Research')
  })

  it('falls back to name when displayName is empty', () => {
    expect(deriveProviderShape([{ name: 'basic', displayName: '', supportsPassword: true }]).providerLabel).toBe(
      'basic'
    )
  })
})

describe('signInLabel', () => {
  it('password gateway gets the plain "Sign in to remote gateway" copy', () => {
    expect(signInLabel({ url: 'x', isPassword: true, providerLabel: 'Username & Password' })).toBe(
      'Sign in to remote gateway'
    )
  })

  it('OAuth gateway names the provider', () => {
    expect(signInLabel({ url: 'x', isPassword: false, providerLabel: 'Nous Research' })).toBe(
      'Sign in with Nous Research'
    )
  })

  it('null reauth falls back to the generic provider phrase', () => {
    expect(signInLabel(null)).toBe('Sign in with your identity provider')
  })
})
