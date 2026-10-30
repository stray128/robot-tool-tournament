/**
 * The attributes of the hosted runner panel's API key field.
 *
 * It is a password field. A text field drawn as dots (`-webkit-text-security`) was tried and dropped: it keeps
 * password managers away, but a screen reader is given the typed characters, where a password field gives it
 * bullets. Exposing the key to assistive technology is worse than a browser offering to save it in the
 * visitor's own password manager, so the key is typed into a plain password field, and the field asks, with
 * what it can, not to be offered to anything else:
 *   - autocomplete off (not `new-password`, which makes browsers offer to generate a password), and
 *     autocorrect, autocapitalize and spell check off (a spell checker can send what it checks to a server);
 *   - the opt-out markers of 1Password, LastPass, Bitwarden and Dashlane, and of Grammarly and LanguageTool
 *     (extensions that read fields). A browser's own password manager has no such marker.
 *
 * The key lives only in the controller's memory (see controller.ts): the field has no name or id, is not in a
 * form, and its value is never written to storage, the URL, history or the console.
 */

/** Spread onto the key input; the value, the change handler, the class and the label are the panel's. */
export const KEY_FIELD_ATTRIBUTES = {
  type: "password",
  autoComplete: "off",
  autoCorrect: "off",
  autoCapitalize: "off",
  spellCheck: false,
  // password managers
  "data-1p-ignore": true,
  "data-lpignore": "true",
  "data-bwignore": true,
  "data-form-type": "other",
  // extensions that read fields
  "data-gramm": "false",
  "data-gramm_editor": "false",
  "data-enable-grammarly": "false",
  "data-lt-active": "false",
} as const;
