import './vendor/dashtreme/bootstrap.min.css'
import './vendor/dashtreme/bootstrap-extended.css'
import './vendor/dashtreme/app.css'
import './vendor/dashtreme/boxicons.css'
import './theme/fonts.css'
import './theme/brand.css'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
