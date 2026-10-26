/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          bg: '#0a0f1d',
          card: '#111827',
          border: '#1f293d',
        },
        accent: {
          cyan: '#00f2fe',
          blue: '#4facfe',
        }
      }
    },
  },
  plugins: [],
}
