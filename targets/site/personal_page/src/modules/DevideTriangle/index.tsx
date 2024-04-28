import React, { useEffect, useRef } from "react"
import useCanvasSize from "@/hooks/useCanvasSize"
import useWebGPU from "@/hooks/useWebGPU";

import styles from './index.scss';
import vert from '@/shaders/devide_triangle/vert.wgsl'
import frag from '@/shaders/devide_triangle/frag.wgsl'
import { vec3, mix } from '@/utils/vec3'

const DevideTriangle = () => {
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

        const points: vec3[] = [];
        function triangle(a: vec3, b: vec3, c: vec3) {
            points.push(a);
            points.push(b);
            points.push(c);
        };
        const vertices: vec3[] = []
        const a = new vec3(-0.5, -0.5, 0.0);
        const b = new vec3(0.5, -0.5, 0.0);
        const c = new vec3(0.0, 0.5, 0.0);
        vertices.push(a);
        vertices.push(b);
        vertices.push(c);
        function devideTriangle(a: vec3, b: vec3, c: vec3, count: number) {
            if (count === 0) {
                triangle(a, b, c);
            }
            else {
                const ab = mix(a, b, 0.5);
                const ac = mix(a, c, 0.5);
                const bc = mix(b, c, 0.5);
                const new_count = count - 1;
                devideTriangle(a, ab, ac, new_count);
                devideTriangle(b, bc, ab, new_count);
                devideTriangle(c, ac, bc, new_count);
            }
        }

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

        devideTriangle(vertices[0], vertices[1], vertices[2], 5);
        const position = new Float32Array(points.length * 3);
        for (let i = 0; i < points.length; i++) {
            position[i * 3 + 0] = points[i].x;
            position[i * 3 + 1] = points[i].y;
            position[i * 3 + 2] = points[i].z;
        }
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

        // passEncoder.draw(3, 1, 0, 0)
        passEncoder.draw(points.length, 1, 0, 0)
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

export default DevideTriangle;