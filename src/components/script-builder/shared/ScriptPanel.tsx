'use client';

import { Card, CardContent, CardHeader } from '../../ui/card';
import { FileText, Copy, Download, Edit3, Scissors, Plus, RotateCcw, Trash2, Zap, Undo, Redo, Layers, Target } from 'lucide-react';
import { Button } from '../../ui/button';
import { useState, useRef, useCallback, useEffect } from 'react';
import { useBulkSelection, BulkSelection } from '../../../hooks/useBulkSelection';
import { BulkModificationPreview } from './BulkModificationPreview';

interface ScriptPanelProps {
  script: string;
  wordCount: number;
  bulletPoints?: any[]; // Legacy prop for backward compatibility
  className?: string;
  sessionId?: string;
  onScriptUpdate?: (newScript: string) => void;
  isModificationMode?: boolean; // New prop to enable/disable modification features
  onScriptHistoryUpdate?: (history: string[], currentIndex: number) => void;
}

interface TextSelection {
  text: string;
  startOffset: number;
  endOffset: number;
  contextBefore: string;
  contextAfter: string;
}

interface ModificationPopup {
  visible: boolean;
  x: number;
  y: number;
  selection: TextSelection | null;
}

export function ScriptPanel({ 
  script, 
  wordCount, 
  bulletPoints, 
  className = '', 
  sessionId,
  onScriptUpdate,
  isModificationMode = true,
  onScriptHistoryUpdate
}: ScriptPanelProps) {
  const [modificationPopup, setModificationPopup] = useState<ModificationPopup>({
    visible: false,
    x: 0,
    y: 0,
    selection: null
  });
  const [isModifying, setIsModifying] = useState(false);
  const [modificationPreview, setModificationPreview] = useState<{
    visible: boolean;
    originalText: string;
    modifiedText: string;
    modificationType: string;
  } | null>(null);
  const [modificationError, setModificationError] = useState<string | null>(null);
  const [scriptHistory, setScriptHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const scriptRef = useRef<HTMLDivElement>(null);

  // Bulk selection functionality
  const bulkSelection = useBulkSelection(scriptRef, { maxSelections: 10 });
  const [bulkModificationPreview, setBulkModificationPreview] = useState<{
    visible: boolean;
    modificationType: string;
    results: any[];
  } | null>(null);
  const [isBulkModifying, setIsBulkModifying] = useState(false);

  // Initialize history when script changes
  useEffect(() => {
    if (script && scriptHistory.length === 0) {
      setScriptHistory([script]);
      setHistoryIndex(0);
    }
  }, [script, scriptHistory.length]);

  const handleCopyScript = async () => {
    try {
      let contentToCopy = script;
      
      // If no script yet but we have bullet points, copy the bullet points
      if (!script && bulletPoints && bulletPoints.length > 0) {
        contentToCopy = bulletPoints.map((point: any, index: number) => 
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
      contentToDownload = bulletPoints.map((point: any, index: number) => 
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

  const getTextSelection = useCallback((): TextSelection | null => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;

    const range = selection.getRangeAt(0);
    const selectedText = selection.toString().trim();
    
    if (!selectedText || !scriptRef.current) return null;

    // Get the full text content
    const fullText = scriptRef.current.textContent || '';
    
    // Find the position of selected text in the full text with better accuracy
    let startOffset = -1;
    let endOffset = -1;
    
    // Try to find the exact position by checking if the range is within our script element
    try {
      if (scriptRef.current.contains(range.commonAncestorContainer)) {
        // Create a range from the start of the script to the start of selection
        const preRange = document.createRange();
        preRange.setStart(scriptRef.current, 0);
        preRange.setEnd(range.startContainer, range.startOffset);
        startOffset = preRange.toString().length;
        endOffset = startOffset + selectedText.length;
      } else {
        // Fallback to indexOf method
        startOffset = fullText.indexOf(selectedText);
        endOffset = startOffset + selectedText.length;
      }
    } catch (e) {
      // Fallback to indexOf method
      startOffset = fullText.indexOf(selectedText);
      endOffset = startOffset + selectedText.length;
    }
    
    if (startOffset === -1) return null;
    
    // Get context before and after with word boundaries
    const contextLength = 150;
    let contextBefore = fullText.substring(Math.max(0, startOffset - contextLength), startOffset);
    let contextAfter = fullText.substring(endOffset, Math.min(fullText.length, endOffset + contextLength));
    
    // Trim to word boundaries for better context
    contextBefore = contextBefore.replace(/^\S*\s/, ''); // Remove partial word at start
    contextAfter = contextAfter.replace(/\s\S*$/, ''); // Remove partial word at end

    return {
      text: selectedText,
      startOffset,
      endOffset,
      contextBefore,
      contextAfter
    };
  }, []);

  const handleTextSelection = useCallback((event: React.MouseEvent) => {
    if (!isModificationMode || !script) return;

    // Handle bulk selection mode
    if (bulkSelection.isBulkMode) {
      bulkSelection.handleBulkTextSelection(event);
      return;
    }

    // Handle single selection mode
    setTimeout(() => {
      const selection = getTextSelection();
      
      if (selection && selection.text.length > 5) {
        const range = window.getSelection()?.getRangeAt(0);
        if (range) {
          const rect = range.getBoundingClientRect();
          const containerRect = scriptRef.current?.getBoundingClientRect();
          
          if (rect && containerRect) {
            // Calculate position relative to viewport with better positioning
            let x = rect.left + rect.width / 2;
            let y = rect.top - 10;
            
            // Ensure popup stays within viewport bounds
            const popupWidth = 300;
            const popupHeight = 50;
            
            if (x + popupWidth / 2 > window.innerWidth) {
              x = window.innerWidth - popupWidth / 2 - 10;
            }
            if (x - popupWidth / 2 < 0) {
              x = popupWidth / 2 + 10;
            }
            if (y < popupHeight) {
              y = rect.bottom + 10;
            }
            
            setModificationPopup({
              visible: true,
              x,
              y,
              selection
            });
          }
        }
      } else {
        setModificationPopup(prev => ({ ...prev, visible: false }));
      }
    }, 10);
  }, [isModificationMode, script, getTextSelection, bulkSelection]);

  const handleModifyText = async (modificationType: string) => {
    if (!modificationPopup.selection || !sessionId) return;

    setIsModifying(true);
    setModificationError(null);
    setModificationPopup(prev => ({ ...prev, visible: false }));

    try {
      // First, get the modification preview
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('action', 'modify_text');
      formData.append('selected_text', modificationPopup.selection.text);
      formData.append('modification_type', modificationType);
      formData.append('context_before', modificationPopup.selection.contextBefore);
      formData.append('context_after', modificationPopup.selection.contextAfter);

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Modification failed');
      }

      const result = await response.json();
      
      // Show preview first
      setModificationPreview({
        visible: true,
        originalText: result.original_text,
        modifiedText: result.modified_text,
        modificationType
      });

    } catch (error) {
      console.error('Text modification failed:', error);
      setModificationError(error instanceof Error ? error.message : 'Modification failed');
    } finally {
      setIsModifying(false);
    }
  };

  const handleApplyModification = async () => {
    if (!modificationPreview || !sessionId) return;

    setIsModifying(true);
    setModificationError(null);

    try {
      const applyFormData = new FormData();
      applyFormData.append('session_id', sessionId);
      applyFormData.append('action', 'apply_modification');
      applyFormData.append('original_text', modificationPreview.originalText);
      applyFormData.append('modified_text', modificationPreview.modifiedText);

      const applyResponse = await fetch('/api/process', {
        method: 'POST',
        body: applyFormData
      });

      if (!applyResponse.ok) {
        const errorData = await applyResponse.json();
        throw new Error(errorData.error || 'Failed to apply modification');
      }

      const applyResult = await applyResponse.json();
      
      // Update the script in the parent component
      if (onScriptUpdate) {
        onScriptUpdate(applyResult.updated_script);
      }

      // Add to history
      const newHistory = scriptHistory.slice(0, historyIndex + 1);
      newHistory.push(applyResult.updated_script);
      setScriptHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
      
      if (onScriptHistoryUpdate) {
        onScriptHistoryUpdate(newHistory, newHistory.length - 1);
      }

      // Clear preview
      setModificationPreview(null);

    } catch (error) {
      console.error('Apply modification failed:', error);
      setModificationError(error instanceof Error ? error.message : 'Failed to apply modification');
    } finally {
      setIsModifying(false);
    }
  };

  const handleCancelModification = () => {
    setModificationPreview(null);
    setModificationError(null);
  };

  // Bulk modification handlers
  const handleBulkModifyText = async (modificationType: string) => {
    if (!bulkSelection.hasSelections || !sessionId) return;

    setIsBulkModifying(true);
    setModificationError(null);

    try {
      const selectionsData = bulkSelection.getBulkSelectionData();
      
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('action', 'modify_bulk_text');
      formData.append('selections', JSON.stringify(selectionsData));
      formData.append('modification_type', modificationType);

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Bulk modification failed');
      }

      const result = await response.json();
      
      // Show bulk preview
      setBulkModificationPreview({
        visible: true,
        modificationType,
        results: result.results
      });

    } catch (error) {
      console.error('Bulk text modification failed:', error);
      setModificationError(error instanceof Error ? error.message : 'Bulk modification failed');
    } finally {
      setIsBulkModifying(false);
    }
  };

  const handleApplyBulkModification = async (selectedResults: any[]) => {
    if (!sessionId) return;

    setIsBulkModifying(true);
    setModificationError(null);

    try {
      const formData = new FormData();
      formData.append('session_id', sessionId);
      formData.append('action', 'apply_bulk_modification');
      formData.append('modifications', JSON.stringify(selectedResults));

      const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to apply bulk modifications');
      }

      const result = await response.json();
      
      // Update the script in the parent component
      if (onScriptUpdate) {
        onScriptUpdate(result.updated_script);
      }

      // Add to history
      const newHistory = scriptHistory.slice(0, historyIndex + 1);
      newHistory.push(result.updated_script);
      setScriptHistory(newHistory);
      setHistoryIndex(newHistory.length - 1);
      
      if (onScriptHistoryUpdate) {
        onScriptHistoryUpdate(newHistory, newHistory.length - 1);
      }

      // Clear bulk selections and preview
      bulkSelection.clearBulkSelections();
      setBulkModificationPreview(null);

    } catch (error) {
      console.error('Apply bulk modification failed:', error);
      setModificationError(error instanceof Error ? error.message : 'Failed to apply bulk modifications');
    } finally {
      setIsBulkModifying(false);
    }
  };

  const handleCancelBulkModification = () => {
    setBulkModificationPreview(null);
    setModificationError(null);
  };

  const handleClickOutside = useCallback((event: React.MouseEvent) => {
    if (modificationPopup.visible) {
      setModificationPopup(prev => ({ ...prev, visible: false }));
    }
  }, [modificationPopup.visible]);

  // Keyboard shortcuts for modification actions
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isModificationMode) return;

      // Global shortcuts
      if (event.ctrlKey || event.metaKey) {
        switch (event.key.toLowerCase()) {
          case 'b':
            event.preventDefault();
            bulkSelection.toggleBulkMode();
            break;
          case 'a':
            if (bulkSelection.isBulkMode) {
              event.preventDefault();
              bulkSelection.selectAllParagraphs();
            }
            break;
        }
      }

      // Single selection shortcuts (when popup is visible)
      if (modificationPopup.visible && modificationPopup.selection) {
        if (event.ctrlKey || event.metaKey) {
          switch (event.key) {
            case '1':
              event.preventDefault();
              handleModifyText('shorten');
              break;
            case '2':
              event.preventDefault();
              handleModifyText('expand');
              break;
            case '3':
              event.preventDefault();
              handleModifyText('rewrite');
              break;
            case '4':
              event.preventDefault();
              handleModifyText('make_engaging');
              break;
            case 'Backspace':
            case 'Delete':
              event.preventDefault();
              handleModifyText('delete');
              break;
          }
        }
      }

      // Bulk selection shortcuts (when bulk mode is active and has selections)
      if (bulkSelection.isBulkMode && bulkSelection.hasSelections) {
        if (event.ctrlKey || event.metaKey) {
          switch (event.key) {
            case '1':
              event.preventDefault();
              handleBulkModifyText('shorten');
              break;
            case '2':
              event.preventDefault();
              handleBulkModifyText('expand');
              break;
            case '3':
              event.preventDefault();
              handleBulkModifyText('rewrite');
              break;
            case '4':
              event.preventDefault();
              handleBulkModifyText('make_engaging');
              break;
            case 'Backspace':
            case 'Delete':
              event.preventDefault();
              handleBulkModifyText('delete');
              break;
          }
        }
      }

      // Escape to close popups
      if (event.key === 'Escape') {
        setModificationPopup(prev => ({ ...prev, visible: false }));
        setModificationPreview(null);
        setBulkModificationPreview(null);
        setModificationError(null);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isModificationMode, modificationPopup.visible, modificationPopup.selection, bulkSelection.isBulkMode, bulkSelection.hasSelections]);

  const ModificationActions = [
    { id: 'shorten', label: 'Shorten', icon: Scissors, description: 'Make it more concise', shortcut: 'Ctrl+1' },
    { id: 'expand', label: 'Expand', icon: Plus, description: 'Add more detail', shortcut: 'Ctrl+2' },
    { id: 'rewrite', label: 'Rewrite', icon: Edit3, description: 'Improve the text', shortcut: 'Ctrl+3' },
    { id: 'make_engaging', label: 'Engaging', icon: Zap, description: 'Make it more compelling', shortcut: 'Ctrl+4' },
    { id: 'delete', label: 'Delete', icon: Trash2, description: 'Remove this text', shortcut: 'Ctrl+Del' },
  ];

  return (
    <div className="relative">
      <Card className={`h-full flex flex-col ${className}`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-sm font-medium text-foreground">Script</h3>
              {isModificationMode && script && (
                <div className="flex items-center gap-2">
                  {/* Bulk Mode Toggle */}
                  <Button
                    variant={bulkSelection.isBulkMode ? "default" : "outline"}
                    size="sm"
                    onClick={() => bulkSelection.toggleBulkMode()}
                    className="h-7 px-2 text-xs"
                    title="Toggle bulk selection mode (Ctrl+B)"
                  >
                    <Layers className="h-3 w-3 mr-1" />
                    Bulk {bulkSelection.hasSelections && `(${bulkSelection.selectedCount})`}
                  </Button>
                  
                  {/* Mode indicator */}
                  <span className="text-xs text-muted-foreground bg-muted px-2 py-1 rounded">
                    {bulkSelection.isBulkMode 
                      ? bulkSelection.hasSelections 
                        ? `${bulkSelection.selectedCount} selections • Ctrl+Click to add more`
                        : "Ctrl+Click to select multiple texts"
                      : "Highlight text to modify"
                    }
                  </span>
                  
                  {/* Quick actions for bulk mode */}
                  {bulkSelection.isBulkMode && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={bulkSelection.selectAllParagraphs}
                      className="h-7 px-2 text-xs"
                      title="Select all paragraphs (Ctrl+A)"
                    >
                      <Target className="h-3 w-3 mr-1" />
                      All
                    </Button>
                  )}
                  
                  <div className="group relative">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-5 w-5 p-0 text-muted-foreground hover:text-foreground"
                    >
                      ?
                    </Button>
                    <div className="absolute left-0 top-6 z-50 hidden group-hover:block w-72 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 text-xs">
                      <h4 className="font-medium mb-2">How to modify text:</h4>
                      <div className="space-y-2">
                        <div>
                          <strong>Single Mode:</strong>
                          <ul className="space-y-1 text-muted-foreground ml-2">
                            <li>• Select any text in the script</li>
                            <li>• Choose from 5 modification options</li>
                            <li>• Preview changes before applying</li>
                          </ul>
                        </div>
                        <div>
                          <strong>Bulk Mode (Ctrl+B):</strong>
                          <ul className="space-y-1 text-muted-foreground ml-2">
                            <li>• Ctrl+Click to select multiple texts</li>
                            <li>• Apply same modification to all</li>
                            <li>• Ctrl+A to select all paragraphs</li>
                          </ul>
                        </div>
                        <div>
                          <strong>Shortcuts:</strong>
                          <ul className="space-y-1 text-muted-foreground ml-2">
                            <li>• Ctrl+1-4 for quick actions</li>
                            <li>• Ctrl+Del to delete</li>
                            <li>• Esc to cancel</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              {/* Undo/Redo buttons */}
              {isModificationMode && script && (
                <>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      if (historyIndex > 0) {
                        const newIndex = historyIndex - 1;
                        setHistoryIndex(newIndex);
                        if (onScriptUpdate) {
                          onScriptUpdate(scriptHistory[newIndex]);
                        }
                      }
                    }}
                    disabled={historyIndex <= 0}
                    className="h-8 w-8 p-0"
                    title="Undo (Ctrl+Z)"
                  >
                    <Undo className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      if (historyIndex < scriptHistory.length - 1) {
                        const newIndex = historyIndex + 1;
                        setHistoryIndex(newIndex);
                        if (onScriptUpdate) {
                          onScriptUpdate(scriptHistory[newIndex]);
                        }
                      }
                    }}
                    disabled={historyIndex >= scriptHistory.length - 1}
                    className="h-8 w-8 p-0"
                    title="Redo (Ctrl+Y)"
                  >
                    <Redo className="h-4 w-4" />
                  </Button>
                </>
              )}
              
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
        
        <CardContent className="flex-1 pt-0" onClick={handleClickOutside}>
          <div className="h-full">
            {script ? (
              <div className="h-full overflow-y-auto">
                <div 
                  ref={scriptRef}
                  className="prose prose-sm dark:prose-invert max-w-none"
                  onMouseUp={handleTextSelection}
                  style={{ userSelect: isModificationMode ? 'text' : 'auto' }}
                >
                  <div className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground bg-transparent border-none p-0 cursor-text">
                    {script}
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full overflow-y-auto">
                {bulletPoints && bulletPoints.length > 0 ? (
                  <div className="prose prose-sm dark:prose-invert max-w-none">
                    <pre className="whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground bg-transparent border-none p-0">
                      {bulletPoints.map((point: any, index: number) => 
                        `${index + 1}. ${point.title}\n${point.description}\n\n`
                      ).join('')}
                    </pre>
                    <div className="text-xs text-muted-foreground mt-4 p-3 bg-muted/30 rounded">
                      💡 Generate a full script to start building your content
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex items-center justify-center text-center">
                    <div className="space-y-2">
                      <FileText className="h-12 w-12 text-muted-foreground mx-auto" />
                      <p className="text-sm text-muted-foreground">
                        Your script will appear here once generated
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Extract a YouTube transcript and generate a full script to get started
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Modification Popup */}
      {modificationPopup.visible && modificationPopup.selection && (
        <div
          className="fixed z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-2 animate-in fade-in-0 zoom-in-95 duration-200"
          style={{
            left: `${modificationPopup.x}px`,
            top: `${modificationPopup.y}px`,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div className="flex gap-1">
            {ModificationActions.map((action, index) => (
              <Button
                key={action.id}
                variant="ghost"
                size="sm"
                onClick={() => handleModifyText(action.id)}
                disabled={isModifying}
                className="h-8 px-2 flex flex-col items-center gap-0 text-xs hover:bg-gray-100 dark:hover:bg-gray-700"
                title={`${action.description} (${action.shortcut})`}
              >
                <action.icon className="h-3 w-3" />
                <span className="text-[10px] leading-none">{action.label}</span>
              </Button>
            ))}
          </div>
          {isModifying && (
            <div className="text-xs text-muted-foreground text-center mt-1 flex items-center justify-center gap-1">
              <div className="w-3 h-3 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
              Modifying...
            </div>
          )}
          <div className="text-[10px] text-muted-foreground text-center mt-1">
            Use Ctrl+1-4 or Esc to close
          </div>
        </div>
      )}

      {/* Modification Preview Modal */}
      {modificationPreview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 animate-in fade-in-0 duration-200">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-foreground">
                Preview Modification: {modificationPreview.modificationType.replace('_', ' ').toUpperCase()}
              </h3>
            </div>
            
            <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">Original Text:</h4>
                <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800 rounded p-3 text-sm">
                  {modificationPreview.originalText}
                </div>
              </div>
              
              <div>
                <h4 className="text-sm font-medium text-muted-foreground mb-2">Modified Text:</h4>
                <div className="bg-green-50 dark:bg-green-950 border border-green-200 dark:border-green-800 rounded p-3 text-sm">
                  {modificationPreview.modifiedText}
                </div>
              </div>
            </div>
            
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex gap-2 justify-end">
              <Button
                variant="outline"
                onClick={handleCancelModification}
                disabled={isModifying}
              >
                Cancel
              </Button>
              <Button
                onClick={handleApplyModification}
                disabled={isModifying}
                className="bg-green-600 hover:bg-green-700"
              >
                {isModifying ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                    Applying...
                  </>
                ) : (
                  'Apply Changes'
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Bulk Selection Popup */}
      {bulkSelection.isBulkMode && bulkSelection.hasSelections && !bulkModificationPreview?.visible && (
        <div className="fixed bottom-4 left-4 z-40 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg p-3 animate-in slide-in-from-bottom-2 duration-200">
          <div className="flex items-center gap-3">
            <div className="text-sm font-medium text-foreground">
              {bulkSelection.selectedCount} selections
            </div>
            <div className="flex gap-1">
              {ModificationActions.map((action) => (
                <Button
                  key={action.id}
                  variant="ghost"
                  size="sm"
                  onClick={() => handleBulkModifyText(action.id)}
                  disabled={isBulkModifying}
                  className="h-7 px-2 text-xs"
                  title={`${action.description} all selections (${action.shortcut})`}
                >
                  <action.icon className="h-3 w-3 mr-1" />
                  {action.label}
                </Button>
              ))}
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={bulkSelection.clearBulkSelections}
              className="h-7 px-2 text-xs text-muted-foreground"
              title="Clear all selections"
            >
              Clear
            </Button>
          </div>
          {isBulkModifying && (
            <div className="text-xs text-muted-foreground text-center mt-2 flex items-center justify-center gap-1">
              <div className="w-3 h-3 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
              Processing {bulkSelection.selectedCount} selections...
            </div>
          )}
        </div>
      )}

      {/* Bulk Modification Preview */}
      {bulkModificationPreview && (
        <BulkModificationPreview
          visible={bulkModificationPreview.visible}
          modificationType={bulkModificationPreview.modificationType}
          results={bulkModificationPreview.results}
          isApplying={isBulkModifying}
          onApply={handleApplyBulkModification}
          onCancel={handleCancelBulkModification}
        />
      )}

      {/* Error Toast */}
      {modificationError && (
        <div className="fixed bottom-4 right-4 z-50 bg-red-500 text-white px-4 py-2 rounded-lg shadow-lg animate-in slide-in-from-bottom-2 duration-300">
          <div className="flex items-center gap-2">
            <span className="text-sm">{modificationError}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setModificationError(null)}
              className="h-6 w-6 p-0 text-white hover:bg-red-600"
            >
              ×
            </Button>
          </div>
        </div>
      )}
    </div>
  );
} 