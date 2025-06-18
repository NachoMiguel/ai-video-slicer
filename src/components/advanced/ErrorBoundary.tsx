'use client';

import { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from '../ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: any) => void;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    console.error('Error Boundary caught an error:', error, errorInfo);
    
    this.setState({
      error,
      errorInfo,
    });

    // Call optional error handler
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // In production, you might want to send this to an error reporting service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry, LogRocket, etc.
      // errorReportingService.captureException(error, { extra: errorInfo });
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  handleReportBug = () => {
    const errorReport = {
      error: this.state.error?.message,
      stack: this.state.error?.stack,
      userAgent: navigator.userAgent,
      url: window.location.href,
      timestamp: new Date().toISOString(),
    };

    // Copy error report to clipboard
    navigator.clipboard.writeText(JSON.stringify(errorReport, null, 2)).then(() => {
      alert('Error report copied to clipboard. You can paste this when reporting the bug.');
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-6">
          <Card className="max-w-md w-full">
            <CardHeader className="text-center">
              <div className="mx-auto w-12 h-12 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="h-6 w-6 text-red-600 dark:text-red-400" />
              </div>
              <CardTitle className="text-xl font-semibold text-foreground">
                Something went wrong
              </CardTitle>
              <p className="text-muted-foreground">
                An unexpected error occurred while rendering this page.
              </p>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Error Details (in development only) */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg border">
                  <h4 className="text-sm font-medium text-foreground mb-2">Error Details:</h4>
                  <p className="text-xs text-red-600 dark:text-red-400 font-mono break-all">
                    {this.state.error.message}
                  </p>
                  {this.state.error.stack && (
                    <details className="mt-2">
                      <summary className="text-xs text-muted-foreground cursor-pointer">
                        Stack Trace
                      </summary>
                      <pre className="text-xs text-muted-foreground mt-1 whitespace-pre-wrap">
                        {this.state.error.stack}
                      </pre>
                    </details>
                  )}
                </div>
              )}

              {/* Actions */}
              <div className="flex flex-col gap-2">
                <Button
                  onClick={this.handleRetry}
                  className="flex items-center justify-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Try Again
                </Button>
                
                <Button
                  variant="outline"
                  onClick={this.handleGoHome}
                  className="flex items-center justify-center gap-2"
                >
                  <Home className="h-4 w-4" />
                  Go to Home
                </Button>
                
                <Button
                  variant="ghost"
                  onClick={this.handleReportBug}
                  className="flex items-center justify-center gap-2 text-muted-foreground"
                >
                  <Bug className="h-4 w-4" />
                  Report Bug
                </Button>
              </div>

              {/* Tips */}
              <div className="pt-4 border-t border-border">
                <h5 className="text-sm font-medium text-foreground mb-2">Quick fixes to try:</h5>
                <ul className="text-xs text-muted-foreground space-y-1">
                  <li>• Refresh the page</li>
                  <li>• Check your internet connection</li>
                  <li>• Clear your browser cache</li>
                  <li>• Disable browser extensions</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// Hook version for functional components
export function useErrorHandler() {
  const handleError = (error: Error, errorInfo?: any) => {
    console.error('Handled error:', error, errorInfo);
    
    // In production, send to error reporting service
    if (process.env.NODE_ENV === 'production') {
      // errorReportingService.captureException(error, { extra: errorInfo });
    }
  };

  return handleError;
} 