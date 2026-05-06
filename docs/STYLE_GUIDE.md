# Symphovie UI/UX Style Guide

**Version:** 1.0  
**Last Updated:** May 6, 2026  
**Purpose:** Maintain consistent visual design and user experience across all screens

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Color Palette](#color-palette)
3. [Typography](#typography)
4. [Spacing & Layout](#spacing--layout)
5. [Button Styles](#button-styles)
6. [Form Controls](#form-controls)
7. [Component Patterns](#component-patterns)
8. [Animation & Transitions](#animation--transitions)
9. [Accessibility](#accessibility)
10. [Screen-Specific Guidelines](#screen-specific-guidelines)

---

## Design Philosophy

Symphovie follows a **dark, gradient-rich aesthetic** inspired by modern creative tools. The UI prioritizes:

- **Visual hierarchy** through color gradients and shadows
- **Professional polish** with smooth transitions and micro-interactions
- **Screen real estate optimization** - maximizing canvas/grid space
- **Contextual information** - showing details on demand, not by default
- **Musician-friendly workflows** - inspired by DAWs and creative software

---

## Color Palette

### Core Brand Colors

```css
:root {
  /* Backgrounds */
  --color-bg-dark: #181929; /* Main app background */
  --color-bg-card: #23243a; /* Card/panel backgrounds */
  --color-bg-nav: #1a1b2e; /* Navigation bar */

  /* Accent Colors */
  --color-accent: #c084fc; /* Primary purple */
  --color-accent-strong: #a21caf; /* Darker purple for hover */
  --color-accent-pink: #f472b6; /* Secondary pink for gradients */

  /* Text Colors */
  --color-text-main: #fff; /* Primary text */
  --color-text-muted: #bdbdd7; /* Secondary/muted text */

  /* UI Elements */
  --color-card-shadow: 0 2px 16px rgba(64, 0, 128, 0.08);
}
```

### Semantic Colors

```css
/* Success/Primary Actions */
--color-primary: #2563eb;
--color-primary-hover: #1d4ed8;

/* Destructive Actions */
--color-danger: #ef4444;
--color-danger-hover: #dc2626;

/* Warning/Preview */
--color-warning: #fbbf24;
--color-warning-hover: #f59e0b;

/* Success States */
--color-success: #10b981;
```

### Color Usage Guidelines

- **Purple gradients** (`--color-accent` → `--color-accent-pink`) for branding and primary CTAs
- **Blue gradients** (`#3b82f6` → `#2563eb`) for primary actions and composition triggers
- **Amber/Yellow** for preview and non-destructive secondary actions
- **Red gradients** for destructive actions and cancel operations
- **Subtle transparency** (`rgba()`) for borders and overlays: `rgba(192, 132, 252, 0.14)` is standard

---

## Typography

### Font Stack

```css
font-family:
  -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu',
  'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif;
```

### Heading Hierarchy

```css
h1 {
  font-size: 2rem;
  font-weight: 700;
} /* Page titles */
h2 {
  font-size: 1.8rem;
  font-weight: 600;
} /* Section headers */
h3 {
  font-size: 1.3rem;
  font-weight: 600;
} /* Subsection headers */
```

### Body Text Sizes

```css
/* Standard body text */
font-size: 1rem; /* 16px base */

/* Small labels / metadata */
font-size: 0.875rem; /* 14px */

/* Extra small (captions, hints) */
font-size: 0.75rem; /* 12px */

/* Compact UI elements (sidebar, tabs) */
font-size: 0.8rem; /* ~13px */
```

### Font Weight Usage

- **700** (bold): Headings, primary labels, important data
- **600** (semi-bold): Subheadings, button text, section titles
- **500** (medium): Secondary labels, metadata
- **400** (normal): Body text, descriptions

---

## Spacing & Layout

### Spacing Scale

Use consistent spacing units based on **0.25rem (4px) increments**:

```css
--spacing-xs: 0.25rem; /* 4px  - tight gaps */
--spacing-sm: 0.5rem; /* 8px  - compact elements */
--spacing-md: 0.75rem; /* 12px - standard gaps */
--spacing-lg: 1rem; /* 16px - comfortable spacing */
--spacing-xl: 1.5rem; /* 24px - section separation */
--spacing-2xl: 2rem; /* 32px - major divisions */
```

### Layout Patterns

#### Three-Panel Editor Layout

```
┌────────────────────────────────────────────────┐
│ Navigation Bar (64px fixed height)            │
├─────────┬──────────────────────┬───────────────┤
│ Left    │ Center Canvas        │ Right Panel   │
│ Sidebar │ (Grid/Preview)       │ (Mixer/Style) │
│ (280px) │ (Flexible)           │ (360px)       │
└─────────┴──────────────────────┴───────────────┘
```

- **Collapsible sidebars** with smooth transitions (0.22s ease)
- **Sticky navigation** at the top (z-index: 100)
- **Responsive breakpoints**: 1100px, 720px, 560px, 480px

#### Card Grid Layouts

- **Projects**: 4 columns (xl) → 3 (lg) → 2 (md) → 1 (sm)
- **Social Feed**: 3 columns → 2 → 1
- **Gap**: 1.5rem standard

---

## Button Styles

### Primary Button (CTAs)

```css
.btn-primary {
  background: linear-gradient(135deg, var(--color-accent) 0%, #a855f7 100%);
  color: #fff;
  border: none;
  border-radius: 0.5rem;
  padding: 0.55rem 1.25rem;
  font-size: 0.875rem;
  font-weight: 600;
  cursor: pointer;
  transition:
    background 0.15s,
    transform 0.15s;
}

.btn-primary:hover:not(:disabled) {
  background: #a855f7;
  transform: translateY(-1px);
}

.btn-primary:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
```

### Secondary Button (Ghost)

```css
.btn-ghost {
  background: none;
  border: 1px solid rgba(192, 132, 252, 0.25);
  color: var(--color-text-muted);
  border-radius: 0.5rem;
  padding: 0.55rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  transition:
    background 0.15s,
    color 0.15s;
}

.btn-ghost:hover {
  background: rgba(255, 255, 255, 0.05);
  color: #fff;
}
```

### Action Buttons (Composition Controls)

Large, gradient buttons with icons for major actions:

```css
.composition-btn {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.85rem 1.5rem;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.25s ease;
}

/* Preview (Amber) */
.composition-btn--preview {
  background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
  color: white;
}

/* Full Composition (Blue) */
.composition-btn--full {
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
}

/* Cancel/Destructive (Red) */
.composition-btn--cancel {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
  color: white;
}
```

**Hover Effects:**

- `transform: translateY(-2px)`
- Enhanced shadow: `0 6px 20px rgba(color, 0.35)`

### Icon Buttons

Small, icon-only buttons for toolbars:

```css
.icon-btn {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 0.7rem;
  border-radius: 0.5rem;
  border: none;
  background: rgba(15, 23, 42, 0.06);
  color: #0f172a;
  font-size: 0.8rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.icon-btn:hover {
  background: #0f172a;
  color: #f8fafc;
  transform: translateY(-1px);
}
```

### Button Do's and Don'ts

✅ **DO:**

- Use gradients for primary actions
- Include icons for clarity
- Provide `:disabled` states with reduced opacity
- Add subtle hover animations (`translateY(-1px)` or `-2px`)
- Use consistent padding and border-radius

❌ **DON'T:**

- Mix flat and gradient styles in the same context
- Use red/destructive colors for primary actions
- Forget disabled states
- Create buttons smaller than 32px tap target (mobile)

---

## Form Controls

### Text Inputs

```css
.form-input {
  background: #12132a;
  border: 1px solid rgba(192, 132, 252, 0.2);
  border-radius: 0.5rem;
  padding: 0.65rem 1rem;
  color: #fff;
  font-size: 0.9rem;
  outline: none;
  transition: border-color 0.15s;
}

.form-input:focus {
  border-color: var(--color-accent);
}

.form-input::placeholder {
  color: var(--color-text-muted);
}
```

### Selects & Dropdowns

- Match input styling
- Use consistent focus states
- Consider custom dropdowns for complex selections

### Labels

```css
.form-label {
  font-size: 0.78rem;
  color: var(--color-text-muted);
  font-weight: 600;
  margin-bottom: 0.35rem;
  display: block;
}
```

### Checkboxes & Radios

```css
input[type='checkbox'],
input[type='radio'] {
  accent-color: var(--color-accent);
}
```

---

## Component Patterns

### Card Component

```css
.card {
  background: var(--color-bg-card);
  border: 1px solid rgba(192, 132, 252, 0.14);
  border-radius: 0.625rem;
  overflow: hidden;
  transition:
    border-color 0.2s,
    box-shadow 0.2s;
}

.card:hover {
  border-color: var(--color-accent);
  box-shadow: 0 4px 20px rgba(192, 132, 252, 0.12);
}
```

### Collapsible Sections

- Header with chevron icon (rotates 180° when expanded)
- `max-height` transitions for smooth expansion
- Padding transition synchronized with height

### Tabs

Pill-style tabs with rounded backgrounds:

```css
.tab {
  padding: 0.5rem 0.875rem;
  border-radius: 0.5rem;
  border: none;
  background: none;
  color: var(--color-text-muted);
  transition:
    background 0.15s,
    color 0.15s;
}

.tab--active {
  background: rgba(192, 132, 252, 0.18);
  color: var(--color-accent);
}
```

### Modals & Overlays

- **Background:** `rgba(0, 0, 0, 0.7)` overlay
- **Content:** Centered card with `var(--color-bg-card)` background
- **Close button:** Top-right corner, subtle X icon
- **Border radius:** 12px minimum
- **z-index:** 1000 for modals

---

## Animation & Transitions

### Standard Durations

```css
--duration-fast: 0.15s; /* Hover states, color changes */
--duration-normal: 0.22s; /* Layout shifts, panel toggles */
--duration-slow: 0.3s; /* Complex animations, max-height */
```

### Easing Functions

```css
--ease-standard: ease;
--ease-out: ease-out;
--ease-in-out: ease-in-out;
```

### Common Animations

**Hover Lift:**

```css
transform: translateY(-2px);
box-shadow: 0 6px 20px rgba(color, 0.35);
transition:
  transform 0.15s ease,
  box-shadow 0.15s ease;
```

**Button Press:**

```css
:active {
  transform: translateY(0);
  box-shadow: 0 2px 8px rgba(color, 0.25);
}
```

**Chevron Rotation:**

```css
.chevron {
  transform: rotate(0deg);
  transition: transform 0.3s ease;
}
.chevron--expanded {
  transform: rotate(180deg);
}
```

**Shimmer Effect (on hover):**

```css
.btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent,
    rgba(255, 255, 255, 0.3),
    transparent
  );
  transform: translateX(-100%);
  transition: transform 0.6s ease;
}
.btn:hover::before {
  transform: translateX(100%);
}
```

---

## Accessibility

### Focus States

- All interactive elements **must** have visible focus indicators
- Use `outline: 2px solid var(--color-accent)` with `outline-offset: 2px`
- Never use `outline: none` without providing alternative focus styling

### Color Contrast

- Text on dark backgrounds: minimum WCAG AA contrast ratio (4.5:1)
- Critical actions: use both color **and** labels/icons
- Don't rely on color alone to convey information

### Keyboard Navigation

- All buttons must be keyboard-accessible
- Modals should trap focus and close on `Escape`
- Tab order should follow visual hierarchy

### Screen Readers

- Use semantic HTML (`<button>`, `<nav>`, `<main>`, etc.)
- Provide `aria-label` for icon-only buttons
- Use `aria-hidden="true"` for decorative icons

---

## Screen-Specific Guidelines

### Landing Page

- Large hero section with gradient background
- Feature cards with hover effects
- Clear primary CTA (Login/Sign Up)
- Minimal navigation (logo + auth button)

### Projects Screen

- 4-column responsive grid
- Card thumbnails with 16:9 aspect ratio
- Active project indicator badge
- Dropdown menu for project actions (edit, delete)
- "New Project" button in header (purple gradient)

### Editor Screen

**Layout:**

- **Top:** Navigation bar with project name, undo/redo, export/import
- **Left:** Instrument sidebar (collapsible to 64px icon strip)
- **Center:**
  - MIDI info display (compact, collapsed by default)
  - Grid canvas with preview/edit modes
  - Reset Layout button (small, top-right of grid)
- **Right:** Mixer/Style panels (collapsible)

**Grid Area:**

- Maximize canvas space - no large instruction banners
- Tooltips or help icons for guidance instead of persistent text
- Reset Layout button: Small icon button, non-intrusive

**MIDI Info Display:**

- Collapsed by default showing: `filename · X tracks · Ys · Y BPM`
- Expandable to show detailed metadata
- Subtle styling - doesn't dominate screen
- Located above grid canvas

### Composition Section

- Large, gradient action buttons (Preview/Full/Cancel)
- Progress bars with animated fills
- Video player with controls
- Share modal with social sharing options

### Social Feed

- 3-column masonry/grid layout
- Composition cards with video thumbnails
- Like/comment counts with icons
- Author avatars and usernames
- Pagination controls at bottom

---

## Code Style Conventions

### CSS Organization

1. Layout properties (display, position, flex, grid)
2. Box model (width, height, padding, margin)
3. Visual styling (background, border, border-radius)
4. Typography (font-size, font-weight, color)
5. Transitions and animations

### Naming Conventions

**BEM-style for components:**

```css
.component-name {
}
.component-name__element {
}
.component-name--modifier {
}
```

**Example:**

```css
.pm-card {
} /* Project Manager Card */
.pm-card__title {
}
.pm-card__thumb {
}
.pm-card--selected {
}
```

### File Organization

```
src/
  components/
    ComponentName/
      ComponentName.jsx
      ComponentName.css
  App.css           (Global app styles, nav, layout)
  index.css         (CSS reset, root variables)
```

---

## Responsive Design

### Breakpoints

```css
/* Mobile (portrait) */
@media (max-width: 480px) {
}

/* Mobile (landscape) / Small tablet */
@media (max-width: 560px) {
}

/* Tablet */
@media (max-width: 720px) {
}

/* Desktop (medium) */
@media (max-width: 1100px) {
}

/* Desktop (large) - default */
```

### Mobile Considerations

- **Touch targets:** Minimum 44x44px for buttons
- **Font sizes:** Increase to 16px minimum to prevent zoom on iOS
- **Sidebars:** Auto-collapse on mobile
- **Grid columns:** Reduce to 1-2 columns
- **Reduce padding:** Use smaller spacing scale on mobile

---

## Implementation Checklist

When creating new components or screens:

- [ ] Use established color variables from `:root`
- [ ] Follow spacing scale (0.25rem increments)
- [ ] Apply standard border-radius (0.5rem for buttons, 0.625rem for cards)
- [ ] Include hover states with subtle animations
- [ ] Provide `:disabled` styles for buttons
- [ ] Add focus indicators for keyboard navigation
- [ ] Test on mobile breakpoints
- [ ] Use semantic HTML elements
- [ ] Add `aria-label` for icon-only buttons
- [ ] Ensure color contrast meets WCAG AA standards
- [ ] Use gradient backgrounds for primary CTAs
- [ ] Apply consistent transitions (0.15s-0.3s)

---

## Future Enhancements

Considerations for future development:

1. **Dark/Light mode toggle** - Variables already support theming
2. **Customizable accent colors** - User preference for brand color
3. **Animations library** - Reusable Framer Motion components
4. **Component library** - Shared UI primitives (Button, Card, Modal, etc.)
5. **Design tokens** - JSON-based token system for cross-platform consistency

---

## Resources & References

- **Color Tool:** [Coolors.co](https://coolors.co) for palette generation
- **Icons:** Lucide React (current implementation)
- **Gradients:** [UI Gradients](https://uigradients.com) for inspiration
- **Shadows:** [Shadow Generator](https://shadows.brumm.af)
- **Animation:** [Easing Functions](https://easings.net)

---

**Maintained by:** Symphovie Development Team  
**Questions?** Refer to this guide or open a discussion in the repo.
