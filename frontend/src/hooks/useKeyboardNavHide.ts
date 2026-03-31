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

const LAB_VH_VAR = '--lab-vh';

function setLabVh(value: string) {
  document.documentElement.style.setProperty(LAB_VH_VAR, value);
}

/**
 * Attach focus/blur listeners to a chat input ref for mobile keyboard handling.
 *
 * On mobile (hover:none + pointer:coarse), focusing the input:
 *   - Adds 'lab-kb-open' to document.body (hides BottomNav via CSS)
 *   - Listens to visualViewport resize to update --lab-vh so the BenTab
 *     container correctly fills only the visible area above the keyboard
 *
 * On blur, everything is reset.
 * On desktop the hook is a no-op.
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

    const updateVh = () => {
      const vp = window.visualViewport;
      const vh = vp ? `${vp.height}px` : '100dvh';
      setLabVh(vh);
    };

    const handleFocus = () => {
      document.body.classList.add('lab-kb-open');
      // Small delay lets iOS finish animating the keyboard before we read height
      setTimeout(updateVh, 100);
      window.visualViewport?.addEventListener('resize', updateVh);
      window.visualViewport?.addEventListener('scroll', updateVh);
    };

    const handleBlur = () => {
      document.body.classList.remove('lab-kb-open');
      setLabVh('100dvh');
      window.visualViewport?.removeEventListener('resize', updateVh);
      window.visualViewport?.removeEventListener('scroll', updateVh);
    };

    input.addEventListener('focus', handleFocus);
    input.addEventListener('blur', handleBlur);

    return () => {
      input.removeEventListener('focus', handleFocus);
      input.removeEventListener('blur', handleBlur);
      document.body.classList.remove('lab-kb-open');
      setLabVh('100dvh');
      window.visualViewport?.removeEventListener('resize', updateVh);
      window.visualViewport?.removeEventListener('scroll', updateVh);
    };
  }, [inputRef]);
}
