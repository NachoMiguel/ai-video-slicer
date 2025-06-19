'use client';

import { Switch } from '../ui/switch';
import { Button } from '../ui/button';
import { Settings, FileText, Zap } from 'lucide-react';

interface SavedScript {
  id: string;
  title: string;
  word_count: number;
  created_at: string;
  updated_at: string;
  source_url?: string;
  filename: string;
  script_length: number;
}

interface DevModeToggleProps {
  skipMode: boolean;
  onToggleSkipMode: (enabled: boolean) => void;
  availableScripts: SavedScript[];
  selectedScript?: SavedScript | null;
  onSelectScript: (script: SavedScript) => void;
  onLoadScript: () => void;
  isLoading?: boolean;
  className?: string;
}

export function DevModeToggle({
  skipMode,
  onToggleSkipMode,
  availableScripts,
  selectedScript,
  onSelectScript,
  onLoadScript,
  isLoading = false,
  className = ''
}: DevModeToggleProps) {
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch {
      return 'Unknown date';
    }
  };

  return (
    <div className={`bg-card border rounded-lg p-4 ${className}`}>
      <div className="space-y-4">
        {/* Toggle Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium text-foreground">Development Mode</span>
          </div>
          
          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground">
              {skipMode ? 'Skip Script Phase' : 'Normal Mode'}
            </span>
            <Switch
              checked={skipMode}
              onCheckedChange={onToggleSkipMode}
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Skip Mode Content */}
        {skipMode && (
          <div className="space-y-3 pt-3 border-t border-border">
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-yellow-500" />
              <span className="text-sm font-medium text-foreground">Skip to Video Processing</span>
            </div>
            
            <p className="text-xs text-muted-foreground">
              Load a pre-generated script and jump directly to the video processing phase.
            </p>

            {/* Script Selection */}
            {availableScripts.length > 0 ? (
              <div className="space-y-2">
                <label className="text-xs font-medium text-foreground">
                  Select Saved Script ({availableScripts.length} available):
                </label>
                
                <div className="space-y-1 max-h-32 overflow-y-auto border rounded-md p-1 bg-muted/20">
                  {availableScripts.map((script) => (
                    <div
                      key={script.id}
                      className={`p-2 rounded border cursor-pointer transition-all duration-200 ${
                        selectedScript?.id === script.id
                          ? 'bg-primary/10 border-primary shadow-sm'
                          : 'bg-card border-border hover:bg-muted hover:shadow-sm'
                      }`}
                      onClick={() => onSelectScript(script)}
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-1">
                            <FileText className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                            <span className="text-xs font-medium text-foreground truncate">
                              {script.title}
                            </span>
                            {selectedScript?.id === script.id && (
                              <div className="w-2 h-2 bg-primary rounded-full flex-shrink-0" />
                            )}
                          </div>
                          <div className="flex items-center gap-3 mt-1">
                            <span className="text-xs text-muted-foreground">
                              {script.word_count.toLocaleString()} words
                            </span>
                            <span className="text-xs text-muted-foreground">
                              {formatDate(script.created_at)}
                            </span>
                          </div>
                          {script.source_url && (
                            <div className="text-xs text-muted-foreground truncate mt-1">
                              Source: {script.source_url}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Load Button */}
                <Button
                  onClick={onLoadScript}
                  disabled={!selectedScript || isLoading}
                  className="w-full flex items-center gap-2"
                  size="sm"
                >
                  <Zap className="h-4 w-4" />
                  {isLoading ? 'Loading Script...' : 'Load Script & Skip to Processing'}
                </Button>
                
                {selectedScript && (
                  <div className="text-xs text-muted-foreground text-center">
                    Selected: {selectedScript.title} • {selectedScript.word_count.toLocaleString()} words
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-6 border rounded-md bg-muted/10">
                <FileText className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm font-medium text-muted-foreground mb-3">
                  No saved scripts found
                </p>
                <div className="text-xs text-muted-foreground space-y-2">
                  <p className="font-medium">To use skip mode, complete the full workflow first:</p>
                  <ol className="text-left space-y-1 max-w-64 mx-auto">
                    <li>1. Switch to Normal Mode</li>
                    <li>2. Enter YouTube URL</li>
                    <li>3. Generate full script</li>
                    <li>4. Modify script using highlight-to-edit</li>
                    <li>5. Click "Save Draft" button</li>
                    <li>6. Return here to use skip mode</li>
                  </ol>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Normal Mode Info */}
        {!skipMode && (
          <div className="pt-3 border-t border-border">
            <p className="text-xs text-muted-foreground">
              Full script building workflow with YouTube input and AI assistance.
            </p>
          </div>
        )}
      </div>
    </div>
  );
} 