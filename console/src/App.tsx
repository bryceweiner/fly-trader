// Routes: the Streamlit console's paths, kept (the vault runbook links /vault), plus the logs page.
import { BrowserRouter, Route, Routes } from 'react-router'
import { AppProvider } from './context'
import { FeedbackProvider } from './components/feedback'
import { Layout } from './components/Layout'
import Data from './pages/Data'
import KalshiData from './pages/KalshiData'
import KalshiModel from './pages/KalshiModel'
import KalshiOverview from './pages/KalshiOverview'
import KalshiTrades from './pages/KalshiTrades'
import Logs from './pages/Logs'
import Model from './pages/Model'
import NotFound from './pages/NotFound'
import Overview from './pages/Overview'
import Processes from './pages/Processes'
import RH from './pages/RH'
import Safety from './pages/Safety'
import Trades from './pages/Trades'
import Vault from './pages/Vault'

export function App() {
  return (
    <BrowserRouter>
      <FeedbackProvider>
        <AppProvider>
          <Layout>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/trades" element={<Trades />} />
              <Route path="/model" element={<Model />} />
              <Route path="/data" element={<Data />} />
              <Route path="/rh" element={<RH />} />
              <Route path="/kalshi" element={<KalshiOverview />} />
              <Route path="/kalshi-trades" element={<KalshiTrades />} />
              <Route path="/kalshi-model" element={<KalshiModel />} />
              <Route path="/kalshi-data" element={<KalshiData />} />
              <Route path="/processes" element={<Processes />} />
              <Route path="/safety" element={<Safety />} />
              <Route path="/vault" element={<Vault />} />
              <Route path="/logs" element={<Logs />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </AppProvider>
      </FeedbackProvider>
    </BrowserRouter>
  )
}
