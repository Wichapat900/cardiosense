// src/components/Auth.jsx
import { useState } from 'react'
import { supabase } from '../lib/supabase'

export default function Auth() {
  const [mode, setMode] = useState('login') // 'login' | 'signup' | 'reset'
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async () => {
    setLoading(true)
    setError(null)
    setMessage(null)

    try {
      if (mode === 'signup') {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setMessage('Check your email to confirm your account!')
      } else if (mode === 'login') {
        const { error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
      } else if (mode === 'reset') {
        const { error } = await supabase.auth.resetPasswordForEmail(email)
        if (error) throw error
        setMessage('Password reset link sent to your email.')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.iconWrap}>
          <span style={{ fontSize: 32 }}>🏋️</span>
        </div>
        <h1 style={styles.title}>Fitness RPG</h1>
        <p style={styles.subtitle}>
          {mode === 'login' ? 'Sign in to continue your journey' :
           mode === 'signup' ? 'Begin your adventure' :
           'Reset your password'}
        </p>

        <input
          style={styles.input}
          type="email"
          placeholder="Email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSubmit()}
        />

        {mode !== 'reset' && (
          <input
            style={styles.input}
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
          />
        )}

        {error && <p style={styles.error}>{error}</p>}
        {message && <p style={styles.success}>{message}</p>}

        <button
          style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Loading…' :
           mode === 'login' ? 'Sign in' :
           mode === 'signup' ? 'Create account' :
           'Send reset link'}
        </button>

        <div style={styles.links}>
          {mode === 'login' && (
            <>
              <button style={styles.link} onClick={() => setMode('signup')}>
                Create account
              </button>
              <span style={{ color: '#ccc' }}>·</span>
              <button style={styles.link} onClick={() => setMode('reset')}>
                Forgot password
              </button>
            </>
          )}
          {(mode === 'signup' || mode === 'reset') && (
            <button style={styles.link} onClick={() => setMode('login')}>
              Back to sign in
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#0f0f13',
    padding: '1rem',
  },
  card: {
    background: '#1a1a24',
    border: '1px solid #2a2a3a',
    borderRadius: 20,
    padding: '2.5rem 2rem',
    width: '100%',
    maxWidth: 380,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: 12,
  },
  iconWrap: {
    width: 64,
    height: 64,
    borderRadius: 20,
    background: '#1e2a3a',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 4,
  },
  title: {
    color: '#f0ede8',
    fontSize: 26,
    fontWeight: 600,
    margin: 0,
    letterSpacing: '-0.02em',
  },
  subtitle: {
    color: '#6b6b80',
    fontSize: 14,
    margin: '0 0 8px',
    textAlign: 'center',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    background: '#12121a',
    border: '1px solid #2a2a3a',
    borderRadius: 12,
    color: '#f0ede8',
    fontSize: 15,
    outline: 'none',
    boxSizing: 'border-box',
  },
  btn: {
    width: '100%',
    padding: '13px',
    background: '#378ADD',
    border: 'none',
    borderRadius: 12,
    color: '#fff',
    fontSize: 15,
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: 4,
  },
  links: {
    display: 'flex',
    gap: 12,
    alignItems: 'center',
    marginTop: 4,
  },
  link: {
    background: 'none',
    border: 'none',
    color: '#378ADD',
    fontSize: 13,
    cursor: 'pointer',
    padding: 0,
  },
  error: {
    color: '#E24B4A',
    fontSize: 13,
    margin: 0,
    textAlign: 'center',
    background: '#1a1010',
    padding: '8px 12px',
    borderRadius: 8,
    width: '100%',
    boxSizing: 'border-box',
  },
  success: {
    color: '#1D9E75',
    fontSize: 13,
    margin: 0,
    textAlign: 'center',
    background: '#0f1a16',
    padding: '8px 12px',
    borderRadius: 8,
    width: '100%',
    boxSizing: 'border-box',
  },
}
