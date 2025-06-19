'use client';

import { useState, useCallback, useRef } from 'react';

export interface BulkSelection {
  id: string;
  text: string;
  startOffset: number;
  endOffset: number;
  contextBefore: string;
  contextAfter: string;
  isSelected: boolean;
  color: string;
}

export interface UseBulkSelectionOptions {
  maxSelections?: number;
  contextLength?: number;
}

export function useBulkSelection(
  scriptRef: React.RefObject<HTMLDivElement>,
  options: UseBulkSelectionOptions = {}
) {
  const { maxSelections = 10, contextLength = 150 } = options;
  
  const [bulkSelections, setBulkSelections] = useState<BulkSelection[]>([]);
  const [isBulkMode, setIsBulkMode] = useState(false);
  const [isSelecting, setIsSelecting] = useState(false);
  
  const selectionColors = [
    'bg-blue-200 dark:bg-blue-800',
    'bg-green-200 dark:bg-green-800', 
    'bg-yellow-200 dark:bg-yellow-800',
    'bg-purple-200 dark:bg-purple-800',
    'bg-pink-200 dark:bg-pink-800',
    'bg-indigo-200 dark:bg-indigo-800',
    'bg-red-200 dark:bg-red-800',
    'bg-orange-200 dark:bg-orange-800',
    'bg-teal-200 dark:bg-teal-800',
    'bg-cyan-200 dark:bg-cyan-800'
  ];

  const getTextSelection = useCallback(() => {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return null;

    const range = selection.getRangeAt(0);
    const selectedText = selection.toString().trim();
    
    if (!selectedText || !scriptRef.current) return null;

    // Get the full text content
    const fullText = scriptRef.current.textContent || '';
    
    // Find the position of selected text in the full text
    let startOffset = -1;
    let endOffset = -1;
    
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
    
    // Get context before and after
    let contextBefore = fullText.substring(Math.max(0, startOffset - contextLength), startOffset);
    let contextAfter = fullText.substring(endOffset, Math.min(fullText.length, endOffset + contextLength));
    
    // Trim to word boundaries for better context
    contextBefore = contextBefore.replace(/^\S*\s/, '');
    contextAfter = contextAfter.replace(/\s\S*$/, '');

    return {
      text: selectedText,
      startOffset,
      endOffset,
      contextBefore,
      contextAfter
    };
  }, [scriptRef, contextLength]);

  const addBulkSelection = useCallback((selectionData: {
    text: string;
    startOffset: number;
    endOffset: number;
    contextBefore: string;
    contextAfter: string;
  }) => {
    if (bulkSelections.length >= maxSelections) {
      console.warn(`Maximum ${maxSelections} selections allowed`);
      return null;
    }

    // Check for overlapping selections
    const hasOverlap = bulkSelections.some(existing => 
      (selectionData.startOffset < existing.endOffset && selectionData.endOffset > existing.startOffset)
    );
    
    if (hasOverlap) {
      console.warn('Overlapping selections are not allowed');
      return null;
    }

    const newSelection: BulkSelection = {
      id: `selection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      text: selectionData.text,
      startOffset: selectionData.startOffset,
      endOffset: selectionData.endOffset,
      contextBefore: selectionData.contextBefore,
      contextAfter: selectionData.contextAfter,
      isSelected: true,
      color: selectionColors[bulkSelections.length % selectionColors.length]
    };

    setBulkSelections(prev => [...prev, newSelection]);
    return newSelection;
  }, [bulkSelections, maxSelections, selectionColors]);

  const removeBulkSelection = useCallback((selectionId: string) => {
    setBulkSelections(prev => prev.filter(s => s.id !== selectionId));
  }, []);

  const toggleBulkSelection = useCallback((selectionId: string) => {
    setBulkSelections(prev => 
      prev.map(s => 
        s.id === selectionId 
          ? { ...s, isSelected: !s.isSelected }
          : s
      )
    );
  }, []);

  const clearBulkSelections = useCallback(() => {
    setBulkSelections([]);
  }, []);

  const handleBulkTextSelection = useCallback((event: React.MouseEvent) => {
    if (!isBulkMode) return;

    // Small delay to ensure selection is complete
    setTimeout(() => {
      const selection = getTextSelection();
      
      if (selection && selection.text.length > 5) {
        // Check if Ctrl/Cmd key is pressed for multi-selection
        if (event.ctrlKey || event.metaKey) {
          const newSelection = addBulkSelection(selection);
          if (newSelection) {
            // Clear the browser selection
            window.getSelection()?.removeAllRanges();
          }
        } else {
          // Single selection in bulk mode - replace all selections
          const newSelection: BulkSelection = {
            id: `selection_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            text: selection.text,
            startOffset: selection.startOffset,
            endOffset: selection.endOffset,
            contextBefore: selection.contextBefore,
            contextAfter: selection.contextAfter,
            isSelected: true,
            color: selectionColors[0]
          };
          
          setBulkSelections([newSelection]);
          window.getSelection()?.removeAllRanges();
        }
      }
    }, 10);
  }, [isBulkMode, getTextSelection, addBulkSelection, selectionColors]);

  const toggleBulkMode = useCallback((enabled?: boolean) => {
    const newMode = enabled !== undefined ? enabled : !isBulkMode;
    setIsBulkMode(newMode);
    
    if (!newMode) {
      clearBulkSelections();
    }
  }, [isBulkMode, clearBulkSelections]);

  const selectAllParagraphs = useCallback(() => {
    if (!scriptRef.current || !isBulkMode) return;

    const fullText = scriptRef.current.textContent || '';
    const paragraphs = fullText.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    
    const newSelections: BulkSelection[] = [];
    let currentOffset = 0;
    
    paragraphs.forEach((paragraph, index) => {
      if (newSelections.length >= maxSelections) return;
      
      const trimmedParagraph = paragraph.trim();
      if (trimmedParagraph.length < 10) return; // Skip very short paragraphs
      
      const startOffset = fullText.indexOf(trimmedParagraph, currentOffset);
      if (startOffset === -1) return;
      
      const endOffset = startOffset + trimmedParagraph.length;
      
      // Get context
      let contextBefore = fullText.substring(Math.max(0, startOffset - contextLength), startOffset);
      let contextAfter = fullText.substring(endOffset, Math.min(fullText.length, endOffset + contextLength));
      
      contextBefore = contextBefore.replace(/^\S*\s/, '');
      contextAfter = contextAfter.replace(/\s\S*$/, '');
      
      newSelections.push({
        id: `paragraph_${index}_${Date.now()}`,
        text: trimmedParagraph,
        startOffset,
        endOffset,
        contextBefore,
        contextAfter,
        isSelected: true,
        color: selectionColors[index % selectionColors.length]
      });
      
      currentOffset = endOffset;
    });
    
    setBulkSelections(newSelections);
  }, [scriptRef, isBulkMode, maxSelections, contextLength, selectionColors]);

  const getSelectedBulkSelections = useCallback(() => {
    return bulkSelections.filter(s => s.isSelected);
  }, [bulkSelections]);

  const getBulkSelectionData = useCallback(() => {
    return bulkSelections.map(selection => ({
      text: selection.text,
      contextBefore: selection.contextBefore,
      contextAfter: selection.contextAfter,
      startOffset: selection.startOffset,
      endOffset: selection.endOffset
    }));
  }, [bulkSelections]);

  return {
    // State
    bulkSelections,
    isBulkMode,
    isSelecting,
    
    // Actions
    toggleBulkMode,
    handleBulkTextSelection,
    addBulkSelection,
    removeBulkSelection,
    toggleBulkSelection,
    clearBulkSelections,
    selectAllParagraphs,
    
    // Getters
    getSelectedBulkSelections,
    getBulkSelectionData,
    
    // Utils
    hasSelections: bulkSelections.length > 0,
    selectedCount: bulkSelections.filter(s => s.isSelected).length,
    maxSelectionsReached: bulkSelections.length >= maxSelections
  };
} 