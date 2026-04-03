// ============================================================================
// ErrorBoundary — catches render-time errors and shows a retry UI.
// React requires class components for error boundaries.
// ============================================================================

import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary] Render error:', error, info.componentStack);
  }

  handleRetry = () => {
    this.setState({ hasError: false });
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          display: 'flex', flexDirection: 'column', alignItems: 'center',
          justifyContent: 'center', height: '100dvh', gap: '16px',
          padding: '24px', textAlign: 'center',
          fontFamily: "'Barlow Condensed', sans-serif",
          background: '#060a0f', color: '#e8edf5',
        }}>
          <span style={{ fontSize: '2rem' }}>📡</span>
          <p style={{ fontSize: '1rem', fontWeight: 700, letterSpacing: '0.05em' }}>
            Something went wrong.
          </p>
          <button
            onClick={this.handleRetry}
            style={{
              padding: '12px 28px', borderRadius: '9999px',
              background: '#14b8a6', color: '#060a0f',
              fontFamily: "'Barlow Condensed', sans-serif",
              fontSize: '0.95rem', fontWeight: 900,
              letterSpacing: '0.08em', textTransform: 'uppercase',
              border: 'none', cursor: 'pointer',
            }}
          >
            Reload App
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
