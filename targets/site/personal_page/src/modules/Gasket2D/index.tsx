import React, { useEffect, useRef } from "react"
import useCanvasSize from "@/hooks/useCanvasSize"
import useWebGPU from "@/hooks/useWebGPU";

import styles from './index.scss';
import vert from '@/shaders/gasket2d/vert.wgsl'
import frag from '@/shaders/gasket2d/frag.wgsl'

const Gasket2D = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null)
    const canvasSize = useCanvasSize();
    const { adapter, device, canvas, context, format } = useWebGPU(canvasRef.current)

    useEffect(() => {
        if (!canvas || !context || !adapter || !device) return

        const canvsConfig: GPUCanvasConfiguration = {
            device,
            format,
            alphaMode: 'opaque'
        }
        context.configure(canvsConfig)

        const createBuffer = (device: GPUDevice, data: any, usage: number) => {
            const buffer = device.createBuffer({
                size: data.byteLength,
                usage,
                mappedAtCreation: true
            });
            const dst = new data.constructor(buffer.getMappedRange());
            dst.set(data);
            buffer.unmap();
            return buffer;
        };
        const position: Float32Array = new Float32Array([
            -0.5, 0.5, 0.0,
            -0.5, -0.5, 0.0,
            0.5, -0.5, 0.0
        ]);
        const positionBuffer = createBuffer(device, position, GPUBufferUsage.VERTEX);

        const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: device.createShaderModule({
                    code: vert
                }),
                entryPoint: 'main',
                buffers: [
                    // positions
                    {
                        arrayStride: 4 * 3,
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: 'float32x3'
                            }
                        ]
                    }
                ]
            },
            fragment: {
                module: device.createShaderModule({
                    code: frag
                }),
                entryPoint: 'main',
                targets: [{ format }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        })
        const commandEncoder = device.createCommandEncoder()
        const textureView = context.getCurrentTexture().createView()
        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }]
        }
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor)
        passEncoder.setPipeline(pipeline)
        passEncoder.setVertexBuffer(0, positionBuffer)
        passEncoder.draw(3, 1, 0, 0)
        passEncoder.end()

        device.queue.submit([commandEncoder.finish()])
    }, [canvasSize, canvas, context, format, adapter, device])

    return (
        <div className={styles.container}>
            <canvas
                ref={canvasRef}
                width={canvasSize.width}
                height={canvasSize.height}
            />
        </div>
    )
}

export default Gasket2D;