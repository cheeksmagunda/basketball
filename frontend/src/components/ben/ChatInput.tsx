// ============================================================================
// ChatInput — Input bar at the bottom of the Ben chat
// Handles text entry, camera upload (base64), and SSE streaming from /api/lab/chat
// ============================================================================

import { useState, useRef, useCallback } from 'react';
import { useLabStore } from '../../store/labStore';
import type { LabMessage } from '../../types';
import styles from './ChatInput.module.css';

interface ChatInputProps {
  inputRef: React.RefObject<HTMLTextAreaElement | null>;
}

// ---------------------------------------------------------------------------
// SSE parser — reads server-sent events from a ReadableStream
// data: {"type":"status","text":"..."} or data: {"type":"content","text":"..."}
// ---------------------------------------------------------------------------

interface SSEEvent {
  type: 'status' | 'content';
  text: string;
}

function parseSSELine(line: string): SSEEvent | null {
  if (!line.startsWith('data: ')) return null;
  try {
    return JSON.parse(line.slice(6)) as SSEEvent;
  } catch {
    return null;
  }
}

export default function ChatInput({ inputRef }: ChatInputProps) {
  const [text, setText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const {
    messages,
    system,
    pendingImage,
    addMessage,
    setPendingImage,
  } = useLabStore();

  // ---------------------------------------------------------------------------
  // Camera / image upload
  // ---------------------------------------------------------------------------

  const handleCameraClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        // Extract base64 and media type from data URL
        const [header, base64] = dataUrl.split(',');
        const mediaType = header.match(/data:([^;]+)/)?.[1] || 'image/png';
        setPendingImage({ base64, mediaType, dataUrl });
      };
      reader.readAsDataURL(file);

      // Reset file input so the same file can be re-selected
      e.target.value = '';
    },
    [setPendingImage],
  );

  const handleRemoveImage = useCallback(() => {
    setPendingImage(null);
  }, [setPendingImage]);

  // ---------------------------------------------------------------------------
  // Send message + SSE streaming
  // ---------------------------------------------------------------------------

  const handleSend = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed && !pendingImage) return;
    if (isSending) return;

    // Build user message
    const userMsg: LabMessage = {
      role: 'user',
      content: trimmed,
      ...(pendingImage ? { imageSrc: pendingImage.dataUrl } : {}),
    };

    addMessage(userMsg);
    setText('');
    setPendingImage(null);
    setIsSending(true);

    // Add a thinking indicator
    const thinkingMsg: LabMessage = {
      role: 'assistant',
      content: '',
      isStatus: true,
    };
    addMessage(thinkingMsg);

    // Build payload: non-hidden messages only
    const chatMessages = [...messages.filter((m) => !m.hidden), userMsg].map(
      (m) => ({
        role: m.role,
        content: m.content,
        ...(m.imageSrc ? { imageSrc: m.imageSrc } : {}),
      }),
    );

    const controller = new AbortController();
    abortRef.current = controller;

    // Connection timeout (60s)
    const timeoutId = setTimeout(() => controller.abort(), 60_000);

    try {
      const response = await fetch('/api/lab/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: chatMessages,
          system: system || undefined,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let accumulatedContent = '';
      let streamMsgAdded = false;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete lines
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;

          const event = parseSSELine(trimmedLine);
          if (!event) continue;

          if (event.type === 'status') {
            // Status events update the thinking label in the bubble
            updateThinkingStatus(event.text);
            continue;
          }

          if (event.type === 'content') {
            accumulatedContent += event.text;

            if (!streamMsgAdded) {
              // Transition the thinking bubble into the first content message
              // in-place — no flash between "Thinking…" disappearing and text appearing
              transitionThinkingToContent(accumulatedContent);
              streamMsgAdded = true;
            } else {
              updateLastAssistantMessage(accumulatedContent);
            }
          }
        }
      }

      // If no content was streamed at all, replace thinking bubble with fallback
      if (!streamMsgAdded) {
        transitionThinkingToContent('No response received.');
      }
    } catch (err: unknown) {
      clearTimeout(timeoutId);

      const message =
        err instanceof DOMException && err.name === 'AbortError'
          ? 'Request timed out. Please try again.'
          : err instanceof Error
            ? err.message
            : 'Something went wrong.';

      // Replace thinking bubble with error message (or append if already gone)
      transitionThinkingToContent(message);
    } finally {
      setIsSending(false);
      abortRef.current = null;
    }
  }, [text, pendingImage, isSending, messages, system, addMessage, setPendingImage]);

  // Helpers to manipulate the lab store messages for streaming

  /** Update the status text inside the thinking bubble while it's still showing */
  const updateThinkingStatus = useCallback((statusText: string) => {
    const store = useLabStore.getState();
    const msgs = [...store.messages];
    const lastIdx = msgs.length - 1;
    if (lastIdx >= 0 && msgs[lastIdx].isStatus) {
      msgs[lastIdx] = { ...msgs[lastIdx], content: statusText };
      useLabStore.setState({ messages: msgs });
    }
  }, []);

  /** Replace the thinking bubble in-place with the first chunk of real content */
  const transitionThinkingToContent = useCallback((content: string) => {
    const store = useLabStore.getState();
    const msgs = [...store.messages];
    const lastIdx = msgs.length - 1;
    if (lastIdx >= 0 && msgs[lastIdx].isStatus) {
      msgs[lastIdx] = { role: 'assistant', content };
      useLabStore.setState({ messages: msgs });
    } else {
      // Thinking message already gone — just append
      useLabStore.setState({
        messages: [...msgs, { role: 'assistant', content }],
      });
    }
  }, []);

  const updateLastAssistantMessage = useCallback((content: string) => {
    const store = useLabStore.getState();
    const msgs = [...store.messages];
    const lastIdx = msgs.length - 1;
    if (lastIdx >= 0 && msgs[lastIdx].role === 'assistant' && !msgs[lastIdx].hidden) {
      msgs[lastIdx] = { ...msgs[lastIdx], content };
      useLabStore.setState({ messages: msgs });
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Key handler: Enter to send, Shift+Enter for newline
  // ---------------------------------------------------------------------------

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className={styles.inputBar}>
      {/* Photo preview */}
      {pendingImage && (
        <div className={styles.previewWrap}>
          <img
            src={pendingImage.dataUrl}
            alt="Pending upload"
            className={styles.previewImg}
          />
          <button
            type="button"
            className={styles.previewRemove}
            onClick={handleRemoveImage}
            aria-label="Remove image"
          >
            &times;
          </button>
        </div>
      )}

      <div className={styles.inputRow}>
        {/* Camera button */}
        <button
          type="button"
          className={styles.cameraBtn}
          onClick={handleCameraClick}
          disabled={isSending}
          aria-label="Attach image"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
        </button>

        {/* Text input */}
        <textarea
          ref={inputRef}
          className={styles.input}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask Ben..."
          rows={1}
          disabled={isSending}
        />

        {/* Send button */}
        <button
          type="button"
          className={`${styles.sendBtn}${isSending ? ` ${styles.sendBtnSending}` : ''}`}
          onClick={handleSend}
          disabled={isSending || (!text.trim() && !pendingImage)}
          aria-label={isSending ? 'Sending…' : 'Send message'}
        >
          {isSending ? (
            <div className={styles.spinner} />
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          )}
        </button>
      </div>

      {/* Hidden file input for camera */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className={styles.hiddenInput}
        onChange={handleFileChange}
      />
    </div>
  );
}
