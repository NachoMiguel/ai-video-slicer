'use client';

import { Card, CardContent, CardHeader } from '../../ui/card';
import { FileText, Copy, Download } from 'lucide-react';
import { Button } from '../../ui/button';

interface ScriptPanelProps {
  script: string;
  wordCount: number;
  bulletPoints?: any[];
  className?: string;
}

export function ScriptPanel({ script, wordCount, bulletPoints, className = '' }: ScriptPanelProps) {
  const handleCopyScript = async () => {
    try {
      await navigator.clipboard.writeText(script);
      // You might want to add a toast notification here
    } catch (err) {
      console.error('Failed to copy script:', err);
    }
  };

  const handleDownloadScript = () => {
    const blob = new Blob([script], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `script-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <Card className={`h-full flex flex-col ${className}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-muted-foreground" />
            <h3 className="text-sm font-medium text-foreground">Script</h3>
          </div>
          <div className="flex items-center gap-2">
            {script && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleCopyScript}
                  className="h-8 w-8 p-0"
                  title="Copy script"
                >
                  <Copy className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDownloadScript}
                  className="h-8 w-8 p-0"
                  title="Download script"
                >
                  <Download className="h-4 w-4" />
                </Button>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="flex-1 pt-0">
        <div className="h-full">
          {script ? (
            <div className="h-full overflow-y-auto">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground bg-transparent border-none p-0">
                  {script}
                </pre>
              </div>
            </div>
          ) : (
            <div className="h-full overflow-y-auto">
              {bulletPoints && bulletPoints.length > 0 ? (
                <div className="space-y-4">
                  <div className="text-center py-4">
                    <FileText className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground mb-2">
                      Script sections ready for generation
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Use the chat to generate content for each section
                    </p>
                  </div>
                  
                  {/* Bullet Points Display */}
                  <div className="space-y-2">
                    {bulletPoints.map((point: any, index: number) => (
                      <div key={point.id || index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg border">
                        <div className="flex-shrink-0 w-6 h-6 bg-primary/10 text-primary rounded-full flex items-center justify-center text-xs font-medium mt-0.5">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium text-foreground">
                            {point.title || `Section ${index + 1}`}
                          </div>
                          <div className="text-xs text-muted-foreground mt-1">
                            {point.description || 'No description'}
                          </div>
                          <div className="flex items-center gap-2 mt-2">
                            <span className="text-xs text-muted-foreground">
                              Target: {point.target_length || 2000} words
                            </span>
                            <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded">
                              {point.emotional_tone || 'neutral'}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                  
                  <div className="text-center text-xs text-muted-foreground bg-muted/30 rounded-lg p-3">
                    💡 Use commands like "/generate section 1" to create content for each section
                  </div>
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-center">
                  <div className="space-y-2">
                    <FileText className="h-12 w-12 text-muted-foreground mx-auto" />
                    <p className="text-sm text-muted-foreground">
                      Your script will appear here as you build it
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Start by generating bullet points or sections using the chat
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
} 