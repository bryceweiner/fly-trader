import { Link } from 'react-router'
import { Callout } from '../components/ui'

export default function NotFound() {
  return <Callout kind="warning">No page here. <Link to="/">Back to the overview</Link>.</Callout>
}
