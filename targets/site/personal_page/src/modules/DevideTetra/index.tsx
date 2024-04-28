import React, { useEffect, useRef } from "react"
import useCanvasSize from "@/hooks/useCanvasSize"
import useWebGPU from "@/hooks/useWebGPU";

import styles from './index.scss';
import vert from '@/shaders/devide_tetra/vert.wgsl'
import frag from '@/shaders/devide_tetra/frag.wgsl'
import { vec3, mix } from '@/utils/vec3'

const DevideTetra = () => {
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
        const colors: vec3[] = [];
        const base_colors = [
            new vec3(1.0, 0.0, 0.5),
            new vec3(0.0, 0.2, 0.5),
            new vec3(0.0, 0.8, 0.5),
            new vec3(0.5, 0.5, 0.5),
        ]
        function triangle(a: vec3, b: vec3, c: vec3, color_idx: number) {
            points.push(a);
            points.push(b);
            points.push(c);
            colors.push(base_colors[color_idx]);
            colors.push(base_colors[color_idx]);
            colors.push(base_colors[color_idx]);
        };
        function tetra(a: vec3, b: vec3, c: vec3, d: vec3) {
            triangle(a, c, b, 0);
            triangle(a, c, d, 1);
            triangle(a, b, d, 2);
            triangle(b, c, d, 3);
        }

        const vertices: vec3[] = []
        const a = new vec3(-0.5, -0.5, 1.0);
        const b = new vec3(0.5, -0.5, 1.0);
        const c = new vec3(0.0, 0.5, 0.5);
        const d = new vec3(0.0, -0.0, 0.0);

        vertices.push(a);
        vertices.push(b);
        vertices.push(c);
        vertices.push(d);

        function devideTetra(a: vec3, b: vec3, c: vec3, d: vec3, count: number) {
            if (count === 0) {
                tetra(a, b, c, d);
            }
            else {
                const ab = mix(a, b, 0.5);
                const ac = mix(a, c, 0.5);
                const ad = mix(a, d, 0.5);
                const bc = mix(b, c, 0.5);
                const bd = mix(b, d, 0.5);
                const cd = mix(c, d, 0.5);
                const new_count = count - 1;
                devideTetra(a, ab, ac, ad, new_count);
                devideTetra(ab, b, bc, bd, new_count);
                devideTetra(ac, bc, c, cd, new_count);
                devideTetra(ad, bd, cd, d, new_count);
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

        devideTetra(vertices[0], vertices[1], vertices[2], vertices[3], 3);

        const position = new Float32Array(points.length * 3);
        for (let i = 0; i < points.length; i++) {
            position[i * 3 + 0] = points[i].x;
            position[i * 3 + 1] = points[i].y;
            position[i * 3 + 2] = points[i].z;
        }
        const positionBuffer = createBuffer(device, position, GPUBufferUsage.VERTEX);

        const color_data = new Float32Array(colors.length * 3);
        for (let i = 0; i < colors.length; i++) {
            color_data[i * 3 + 0] = colors[i].x;
            color_data[i * 3 + 1] = colors[i].y;
            color_data[i * 3 + 2] = colors[i].z;
        }
        const colorBuffer = createBuffer(device, color_data, GPUBufferUsage.VERTEX);

        const newDepthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            sampleCount: 1,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

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
                    },
                    // color
                    {
                        arrayStride: 4 * 3,
                        attributes: [
                            {
                                shaderLocation: 1,
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
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        })
        const commandEncoder = device.createCommandEncoder()
        const textureView = context.getCurrentTexture().createView()
        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store'
            }],
            depthStencilAttachment: {
                view: newDepthTexture.createView(),
                depthClearValue: 1,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        }
        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor)
        passEncoder.setPipeline(pipeline)
        passEncoder.setVertexBuffer(0, positionBuffer)
        passEncoder.setVertexBuffer(1, colorBuffer)
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

export default DevideTetra;