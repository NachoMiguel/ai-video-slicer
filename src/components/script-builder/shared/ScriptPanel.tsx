'use client';

import { Card, CardContent, CardHeader } from '../../ui/card';
import { FileText, Copy, Download } from 'lucide-react';
import { Button } from '../../ui/button';

interface BulletPoint {
  id: string;
  title: string;
  description: string;
  target_length: number;
  importance: string;
  order: number;
  key_points: string[];
  emotional_tone: string;
  engagement_strategy: string;
}

interface ScriptPanelProps {
  script: string;
  wordCount: number;
  bulletPoints?: BulletPoint[];
  className?: string;
}

export function ScriptPanel({ script, wordCount, bulletPoints, className = '' }: ScriptPanelProps) {
  const handleCopyScript = async () => {
    try {
      let contentToCopy = script;
      
      // If no script yet but we have bullet points, copy the bullet points
      if (!script && bulletPoints && bulletPoints.length > 0) {
        contentToCopy = bulletPoints.map((point: BulletPoint, index: number) => 
          `${index + 1}. ${point.title}\n${point.description}\n`
        ).join('\n');
      }
      
      await navigator.clipboard.writeText(contentToCopy);
      // You might want to add a toast notification here
    } catch (err) {
      console.error('Failed to copy content:', err);
    }
  };

  const handleDownloadScript = () => {
    let contentToDownload = script;
    
    // If no script yet but we have bullet points, download the bullet points
    if (!script && bulletPoints && bulletPoints.length > 0) {
      contentToDownload = bulletPoints.map((point: BulletPoint, index: number) => 
        `${index + 1}. ${point.title}\n${point.description}\n`
      ).join('\n');
    }
    
    const blob = new Blob([contentToDownload], { type: 'text/plain' });
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
            {(script || (bulletPoints && bulletPoints.length > 0)) && (
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleCopyScript}
                  className="h-8 w-8 p-0"
                  title={script ? "Copy script" : "Copy bullet points"}
                >
                  <Copy className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDownloadScript}
                  className="h-8 w-8 p-0"
                  title={script ? "Download script" : "Download bullet points"}
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
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground bg-transparent border-none p-0">
                    {bulletPoints.map((point: BulletPoint, index: number) => 
                      `${index + 1}. ${point.title}\n${point.description}\n\n`
                    ).join('')}
                  </pre>
                  <div className="text-xs text-muted-foreground mt-4 p-3 bg-muted/30 rounded">
                    💡 Use commands like "start with point 1" to create content for each section
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