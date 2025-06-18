'use client';

import { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../ui/card';
import { Button } from '../../ui/button';
import { 
  Send, 
  Bot, 
  User, 
  Loader2, 
  MessageSquare, 
  Command, 
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

interface ChatMessage {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  command?: string;
  status?: 'sending' | 'sent' | 'error';
}

interface ChatInterfaceProps {
  sessionId: string;
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  wordCount?: number;
  targetWordCount?: number;
}

const SLASH_COMMANDS = [
  {
    command: '/generate',
    description: 'Generate content for a specific section',
    usage: '/generate [section name]'
  },
  {
    command: '/refine',
    description: 'Refine existing content',
    usage: '/refine [section name] [instructions]'
  },
  {
    command: '/wordcount',
    description: 'Get current word count and progress',
    usage: '/wordcount'
  },
  {
    command: '/help',
    description: 'Show available commands',
    usage: '/help'
  },
  {
    command: '/summary',
    description: 'Get script summary and overview',
    usage: '/summary'
  },
  {
    command: '/tone',
    description: 'Adjust tone for sections',
    usage: '/tone [casual|professional|energetic] [section]'
  }
];

export function ChatInterface({ 
  sessionId, 
  messages, 
  onSendMessage, 
  isLoading = false,
  wordCount = 0,
  targetWordCount = 2000
}: ChatInterfaceProps) {
  const [inputValue, setInputValue] = useState('');
  const [showCommands, setShowCommands] = useState(false);
  const [filteredCommands, setFilteredCommands] = useState(SLASH_COMMANDS);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (value: string) => {
    setInputValue(value);
    
    // Handle slash commands
    if (value.startsWith('/')) {
      const commandQuery = value.slice(1).toLowerCase();
      const filtered = SLASH_COMMANDS.filter(cmd => 
        cmd.command.slice(1).toLowerCase().includes(commandQuery) ||
        cmd.description.toLowerCase().includes(commandQuery)
      );
      setFilteredCommands(filtered);
      setShowCommands(filtered.length > 0);
      setSelectedCommandIndex(0);
    } else {
      setShowCommands(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showCommands) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex(prev => 
          prev < filteredCommands.length - 1 ? prev + 1 : 0
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex(prev => 
          prev > 0 ? prev - 1 : filteredCommands.length - 1
        );
      } else if (e.key === 'Tab') {
        e.preventDefault();
        selectCommand(filteredCommands[selectedCommandIndex]);
      } else if (e.key === 'Escape') {
        setShowCommands(false);
      }
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const selectCommand = (command: any) => {
    setInputValue(command.usage);
    setShowCommands(false);
    textareaRef.current?.focus();
  };

  const handleSend = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
      setShowCommands(false);
    }
  };

  const formatMessage = (content: string) => {
    // Simple markdown-like formatting
    let formatted = content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code class="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded text-sm">$1</code>');
    
    return { __html: formatted };
  };

  const getMessageIcon = (message: ChatMessage) => {
    switch (message.type) {
      case 'user':
        return <User className="h-4 w-4" />;
      case 'assistant':
        return <Bot className="h-4 w-4" />;
      case 'system':
        return <MessageSquare className="h-4 w-4" />;
      default:
        return <MessageSquare className="h-4 w-4" />;
    }
  };

  const getMessageStatusIcon = (message: ChatMessage) => {
    switch (message.status) {
      case 'sending':
        return <Loader2 className="h-3 w-3 animate-spin text-gray-400" />;
      case 'sent':
        return <CheckCircle className="h-3 w-3 text-green-500" />;
      case 'error':
        return <AlertCircle className="h-3 w-3 text-red-500" />;
      default:
        return null;
    }
  };

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.type === 'user';
    const isSystem = message.type === 'system';
    
    return (
      <div
        key={message.id}
        className={`flex gap-3 p-4 ${
          isUser ? 'bg-primary/5' : isSystem ? 'bg-yellow-50 dark:bg-yellow-950' : 'bg-background'
        }`}
      >
        <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser 
            ? 'bg-primary text-primary-foreground' 
            : isSystem
            ? 'bg-yellow-500 text-yellow-50'
            : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'
        }`}>
          {getMessageIcon(message)}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-sm font-medium text-foreground">
              {isUser ? 'You' : isSystem ? 'System' : 'AI Assistant'}
            </span>
            <span className="text-xs text-muted-foreground">
              {message.timestamp.toLocaleTimeString()}
            </span>
            {getMessageStatusIcon(message)}
          </div>
          
          {message.command && (
            <div className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs font-mono mb-2">
              <Command className="h-3 w-3" />
              {message.command}
            </div>
          )}
          
          <div 
            className="text-sm text-foreground leading-relaxed"
            dangerouslySetInnerHTML={formatMessage(message.content)}
          />
        </div>
      </div>
    );
  };

  const progressPercentage = targetWordCount > 0 ? Math.min((wordCount / targetWordCount) * 100, 100) : 0;

  return (
    <Card className="h-[600px] flex flex-col">
      <CardHeader className="flex-shrink-0 pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Interactive Script Builder
          </CardTitle>
          
          {/* Progress Indicator */}
          <div className="flex items-center gap-3 text-sm">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">Progress:</span>
              <div className="w-24 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${progressPercentage}%` }}
                />
              </div>
              <span className="font-medium">{progressPercentage.toFixed(0)}%</span>
            </div>
            <div className="flex items-center gap-1 text-muted-foreground">
              <span>{wordCount}</span>
              <span>/</span>
              <span>{targetWordCount} words</span>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto border-b border-border">
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-center p-8">
              <div className="space-y-4">
                <Bot className="h-12 w-12 text-muted-foreground mx-auto" />
                <div>
                  <h3 className="text-lg font-medium text-foreground">Ready to Build Your Script</h3>
                  <p className="text-muted-foreground mt-1">
                    Start by asking me to generate sections, refine content, or use slash commands for quick actions.
                  </p>
                </div>
                <div className="text-xs text-muted-foreground">
                  Type <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">/help</code> to see available commands
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-0">
              {messages.map(renderMessage)}
              {isLoading && (
                <div className="flex gap-3 p-4 bg-background">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-medium text-foreground">AI Assistant</span>
                      <span className="text-xs text-muted-foreground">thinking...</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Generating response...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Command Suggestions */}
        {showCommands && (
          <div className="border-b border-border bg-gray-50 dark:bg-gray-900 max-h-48 overflow-y-auto">
            <div className="p-2">
              <div className="text-xs text-muted-foreground mb-2 px-2">Available Commands:</div>
              {filteredCommands.map((command, index) => (
                <button
                  key={command.command}
                  className={`w-full text-left p-2 rounded text-sm transition-colors ${
                    index === selectedCommandIndex 
                      ? 'bg-primary text-primary-foreground' 
                      : 'hover:bg-gray-200 dark:hover:bg-gray-700'
                  }`}
                  onClick={() => selectCommand(command)}
                >
                  <div className="font-mono font-medium">{command.command}</div>
                  <div className={`text-xs ${
                    index === selectedCommandIndex 
                      ? 'text-primary-foreground/80' 
                      : 'text-muted-foreground'
                  }`}>
                    {command.description}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 bg-background">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => handleInputChange(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask me to generate content, refine sections, or use /commands..."
                className="w-full resize-none border border-border rounded-lg px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent min-h-[60px] max-h-32"
                disabled={isLoading}
                rows={2}
              />
              
              {/* Send Button */}
              <Button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                size="sm"
                className="absolute right-2 bottom-2 h-8 w-8 p-0"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Send className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          
          {/* Helper Text */}
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <div className="flex items-center gap-4">
              <span>Press Enter to send, Shift+Enter for new line</span>
              <span>Type / for commands</span>
            </div>
            <div className="flex items-center gap-1">
              <Clock className="h-3 w-3" />
              <span>Session: {sessionId.slice(0, 8)}</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
} 