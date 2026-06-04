import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'PyroSense — AI Forest Fire Prediction',
  description: 'AI-powered forest fire prediction and spread simulation dashboard with real-time risk mapping.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        {children}
      </body>
    </html>
  )
}
