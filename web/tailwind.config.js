/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Archivo', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['"Spline Sans Mono"', 'monospace'],
      },
      colors: {
        noct: {
          navy: '#1B1E4A',
          ink: '#0E1330',
          body: '#3A3F66',
          secondary: '#4A4F73',
          panel: '#F1F3F9',
          hairline: '#E3E6F1',
          hairlineDark: '#CBD0E2',
          emerald: '#046A38',
          mint: '#4ADE9C',
          crimson: '#D21319',
          crimsonNavy: '#FF7A7F',
          ochre: '#8A5A00',
          ochreNavy: '#F2B84B',
          navyMuted: '#AEB4E0',
          navyLight: '#C9CEF0',
        },
      },
      borderRadius: {
        'sharp': '2px',
        'panel': '4px',
      },
      maxWidth: {
        'content': '1160px',
      },
    },
  },
  plugins: [],
}
