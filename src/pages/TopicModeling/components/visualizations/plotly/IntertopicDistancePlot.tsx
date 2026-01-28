/**
 * Intertopic Distance Plot Component
 * Uses react-plotly.js to display inter-topic distance map
 */

import { useMemo } from 'react'
import Plot from 'react-plotly.js'
import { Box, CircularProgress, Typography } from '@mui/material'
import { useTranslation } from 'react-i18next'
import { useTheme } from '@mui/material/styles'

interface IntertopicDistancePlotProps {
  data: any // Plotly figure data
  height?: number
}

export default function IntertopicDistancePlot({ data, height = 600 }: IntertopicDistancePlotProps) {
  const { t } = useTranslation()
  const theme = useTheme()

  const plotlyConfig = useMemo(() => {
    return {
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
      responsive: true,
      toImageButtonOptions: {
        format: 'png',
        filename: 'intertopic-distance',
        height,
        width: 800,
        scale: 3
      }
    }
  }, [height])

  const plotlyLayout = useMemo(() => {
    const isDarkMode = theme.palette.mode === 'dark'
    const fontColor = isDarkMode ? '#e0e0e0' : theme.palette.text.primary
    const axisColor = isDarkMode ? '#ccc' : '#666'
    const gridColor = isDarkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)'
    
    if (!data || !data.layout) {
      return {
        autosize: true,
        height,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
        font: {
          color: fontColor,
          family: 'Arial, sans-serif',
          size: 12
        }
      }
    }

    // Handle title - can be string or object
    let titleConfig = data.layout.title
    if (typeof titleConfig === 'string') {
      titleConfig = { text: titleConfig, font: { color: fontColor } }
    } else if (titleConfig) {
      titleConfig = { ...titleConfig, font: { ...titleConfig.font, color: fontColor } }
    }

    // Merge with theme-aware colors - force override title and axis colors
    return {
      ...data.layout,
      autosize: true,
      height,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: isDarkMode ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.02)',
      font: {
        ...data.layout.font,
        color: fontColor
      },
      title: titleConfig,
      xaxis: {
        ...data.layout.xaxis,
        color: axisColor,
        tickfont: { ...data.layout.xaxis?.tickfont, color: axisColor },
        titlefont: { ...data.layout.xaxis?.titlefont, color: axisColor },
        gridcolor: gridColor
      },
      yaxis: {
        ...data.layout.yaxis,
        color: axisColor,
        tickfont: { ...data.layout.yaxis?.tickfont, color: axisColor },
        titlefont: { ...data.layout.yaxis?.titlefont, color: axisColor },
        gridcolor: gridColor
      }
    }
  }, [data, height, theme])

  if (!data) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height }}>
        <CircularProgress />
      </Box>
    )
  }

  if (data.error) {
    return (
      <Box sx={{ p: 2, textAlign: 'center' }}>
        <Typography variant="body2" color="error">
          {data.error}
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ width: '100%', height }}>
      <Plot
        data={data.data || []}
        layout={plotlyLayout}
        config={plotlyConfig}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler
      />
    </Box>
  )
}
