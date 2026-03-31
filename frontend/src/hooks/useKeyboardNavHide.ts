// ============================================================================
// useKeyboardNavHide
// Ben (Lab) tab mobile keyboard management.
//
// On mobile (hover:none and pointer:coarse), when the chat input is focused:
//   - Adds 'lab-kb-open' class to document.body
// On blur:
//   - Removes the class
//
// On desktop the hook is a no-op -- matches the vanilla JS guard:
//   window.matchMedia('(hover: none) and (pointer: coarse)')
// ============================================================================

import { useEffect, useRef } from 'react';

/**
 * Attach focus/blur listeners to a chat input ref for mobile keyboard handling.
 *
 * @param inputRef - React ref to the Lab chat input element
 */
export function useKeyboardNavHide(
  inputRef: React.RefObject<HTMLInputElement | HTMLTextAreaElement | null>,
) {
  const isMobile = useRef(false);

  useEffect(() => {
    isMobile.current = window.matchMedia('(hover: none) and (pointer: coarse)').matches;

    const input = inputRef.current;
    if (!input || !isMobile.current) return;

    const handleFocus = () => {
      document.body.classList.add('lab-kb-open');
    };
    const handleBlur = () => {
      document.body.classList.remove('lab-kb-open');
    };

    input.addEventListener('focus', handleFocus);
    input.addEventListener('blur', handleBlur);

    return () => {
      input.removeEventListener('focus', handleFocus);
      input.removeEventListener('blur', handleBlur);
      document.body.classList.remove('lab-kb-open');
    };
  }, [inputRef]);
}
