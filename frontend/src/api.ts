import { useCallback, useEffect, useRef, useState } from 'react'
import type { DashboardData } from './types'

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers)
  // The browser supplies the multipart boundary; never label FormData as JSON.
  if (!(options.body instanceof FormData) && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(`/api${path}`, { ...options, cache: 'no-store', headers })
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(typeof body?.detail === 'string' ? body.detail : 'The local API is unavailable. Check that FastAPI is running on port 8000.')
  if (!body) throw new Error('The local API returned an unreadable response.')
  return body as T
}

export function useDashboard() {
  const [data, setData] = useState<DashboardData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const refresh = useCallback(async () => {
    controller.current?.abort()
    const current = new AbortController()
    controller.current = current
    setLoading(true)
    setError('')
    try { setData(await request<DashboardData>('/dashboard', { signal: current.signal })) }
    catch (e) { if (!current.signal.aborted) setError(e instanceof Error ? e.message : 'Could not load saved reports.') }
    finally { if (!current.signal.aborted) setLoading(false) }
  }, [])
  useEffect(() => { void refresh(); return () => controller.current?.abort() }, [refresh])
  return { data, loading, error, refresh }
}
