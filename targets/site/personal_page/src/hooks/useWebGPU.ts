import { useEffect, useState } from "react"
import useDevice from "./useDevice"

const useWebGPU = (canvas: HTMLCanvasElement | null | undefined) => {

    const [context, setContext] = useState<GPUCanvasContext>()
    const [format, setFormat] = useState<GPUTextureFormat>('bgra8unorm')
    const { adapter, device, gpu } = useDevice()

    useEffect(() => {

        if (!canvas || !gpu || !adapter) return

        const context = canvas.getContext('webgpu')
        if (context === null) return
        setContext(context)
        setFormat(gpu.getPreferredCanvasFormat()) // bgra8unorm

    }, [canvas, adapter, gpu])

    return { canvas, context, format, adapter, device }
}

export default useWebGPU