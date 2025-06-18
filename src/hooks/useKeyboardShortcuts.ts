'use client';

import { useEffect, useCallback, useRef } from 'react';
import { usePreference } from '../stores/settingsStore';

export interface KeyboardShortcut {
  id: string;
  keys: string[];
  description: string;
  action: () => void;
  category?: string;
  preventDefault?: boolean;
  allowInInputs?: boolean;
}

interface UseKeyboardShortcutsOptions {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
}

export function useKeyboardShortcuts({ shortcuts, enabled = true }: UseKeyboardShortcutsOptions) {
  const keyboardShortcutsEnabled = usePreference('enableKeyboardShortcuts');
  const pressedKeysRef = useRef<Set<string>>(new Set());
  const shortcutsMapRef = useRef<Map<string, KeyboardShortcut>>(new Map());

  // Build shortcuts map
  useEffect(() => {
    shortcutsMapRef.current.clear();
    shortcuts.forEach(shortcut => {
      const keyCombo = shortcut.keys.map(k => k.toLowerCase()).sort().join('+');
      shortcutsMapRef.current.set(keyCombo, shortcut);
    });
  }, [shortcuts]);

  const normalizeKey = useCallback((key: string): string => {
    const keyMap: { [key: string]: string } = {
      ' ': 'space',
      'Control': 'ctrl',
      'Meta': 'cmd',
      'Alt': 'alt',
      'Shift': 'shift',
      'Enter': 'enter',
      'Escape': 'escape',
      'Tab': 'tab',
      'Backspace': 'backspace',
      'Delete': 'delete',
      'ArrowUp': 'up',
      'ArrowDown': 'down',
      'ArrowLeft': 'left',
      'ArrowRight': 'right',
    };

    return keyMap[key] || key.toLowerCase();
  }, []);

  const isInputElement = useCallback((target: EventTarget | null): boolean => {
    if (!target || !(target instanceof HTMLElement)) return false;
    
    const tagName = target.tagName.toLowerCase();
    const isInput = ['input', 'textarea', 'select'].includes(tagName);
    const isContentEditable = target.contentEditable === 'true';
    
    return isInput || isContentEditable;
  }, []);

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabled || !keyboardShortcutsEnabled) return;

    const normalizedKey = normalizeKey(event.key);
    pressedKeysRef.current.add(normalizedKey);

    // Build current key combination
    const currentKeys = Array.from(pressedKeysRef.current).sort().join('+');
    const shortcut = shortcutsMapRef.current.get(currentKeys);

    if (shortcut) {
      // Check if we should ignore shortcuts in input elements
      if (!shortcut.allowInInputs && isInputElement(event.target)) {
        return;
      }

      if (shortcut.preventDefault !== false) {
        event.preventDefault();
      }

      shortcut.action();
    }
  }, [enabled, keyboardShortcutsEnabled, normalizeKey, isInputElement]);

  const handleKeyUp = useCallback((event: KeyboardEvent) => {
    const normalizedKey = normalizeKey(event.key);
    pressedKeysRef.current.delete(normalizedKey);
  }, [normalizeKey]);

  const handleBlur = useCallback(() => {
    // Clear all pressed keys when window loses focus
    pressedKeysRef.current.clear();
  }, []);

  useEffect(() => {
    if (!enabled || !keyboardShortcutsEnabled) return;

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, [enabled, keyboardShortcutsEnabled, handleKeyDown, handleKeyUp, handleBlur]);

  return {
    isEnabled: enabled && keyboardShortcutsEnabled,
    shortcuts: shortcuts.filter(s => enabled && keyboardShortcutsEnabled),
  };
}

// Utility function to format key combinations for display
export function formatKeyCombo(keys: string[]): string {
  const keyMap: { [key: string]: string } = {
    'ctrl': '⌃',
    'cmd': '⌘',
    'alt': '⌥',
    'shift': '⇧',
    'space': '␣',
    'enter': '↵',
    'escape': '⎋',
    'tab': '⇥',
    'backspace': '⌫',
    'delete': '⌦',
    'up': '↑',
    'down': '↓',
    'left': '←',
    'right': '→',
  };

  return keys.map(key => {
    const normalizedKey = key.toLowerCase();
    return keyMap[normalizedKey] || key.toUpperCase();
  }).join(' + ');
}

// Common keyboard shortcuts
export const createCommonShortcuts = (actions: {
  save?: () => void;
  undo?: () => void;
  redo?: () => void;
  copy?: () => void;
  paste?: () => void;
  selectAll?: () => void;
  find?: () => void;
  newItem?: () => void;
  delete?: () => void;
  refresh?: () => void;
  help?: () => void;
  settings?: () => void;
  escape?: () => void;
}): KeyboardShortcut[] => {
  const shortcuts: KeyboardShortcut[] = [];

  if (actions.save) {
    shortcuts.push({
      id: 'save',
      keys: ['ctrl', 's'],
      description: 'Save current work',
      action: actions.save,
      category: 'General',
    });
  }

  if (actions.undo) {
    shortcuts.push({
      id: 'undo',
      keys: ['ctrl', 'z'],
      description: 'Undo last action',
      action: actions.undo,
      category: 'Edit',
    });
  }

  if (actions.redo) {
    shortcuts.push({
      id: 'redo',
      keys: ['ctrl', 'y'],
      description: 'Redo last action',
      action: actions.redo,
      category: 'Edit',
    });
  }

  if (actions.copy) {
    shortcuts.push({
      id: 'copy',
      keys: ['ctrl', 'c'],
      description: 'Copy to clipboard',
      action: actions.copy,
      category: 'Edit',
      allowInInputs: true,
    });
  }

  if (actions.paste) {
    shortcuts.push({
      id: 'paste',
      keys: ['ctrl', 'v'],
      description: 'Paste from clipboard',
      action: actions.paste,
      category: 'Edit',
      allowInInputs: true,
    });
  }

  if (actions.selectAll) {
    shortcuts.push({
      id: 'selectAll',
      keys: ['ctrl', 'a'],
      description: 'Select all',
      action: actions.selectAll,
      category: 'Edit',
      allowInInputs: true,
    });
  }

  if (actions.find) {
    shortcuts.push({
      id: 'find',
      keys: ['ctrl', 'f'],
      description: 'Find in page',
      action: actions.find,
      category: 'Navigation',
    });
  }

  if (actions.newItem) {
    shortcuts.push({
      id: 'new',
      keys: ['ctrl', 'n'],
      description: 'Create new item',
      action: actions.newItem,
      category: 'General',
    });
  }

  if (actions.delete) {
    shortcuts.push({
      id: 'delete',
      keys: ['delete'],
      description: 'Delete selected item',
      action: actions.delete,
      category: 'Edit',
    });
  }

  if (actions.refresh) {
    shortcuts.push({
      id: 'refresh',
      keys: ['f5'],
      description: 'Refresh page',
      action: actions.refresh,
      category: 'General',
    });
  }

  if (actions.help) {
    shortcuts.push({
      id: 'help',
      keys: ['f1'],
      description: 'Show help',
      action: actions.help,
      category: 'General',
    });
  }

  if (actions.settings) {
    shortcuts.push({
      id: 'settings',
      keys: ['ctrl', ','],
      description: 'Open settings',
      action: actions.settings,
      category: 'General',
    });
  }

  if (actions.escape) {
    shortcuts.push({
      id: 'escape',
      keys: ['escape'],
      description: 'Cancel or close',
      action: actions.escape,
      category: 'Navigation',
    });
  }

  return shortcuts;
}; 