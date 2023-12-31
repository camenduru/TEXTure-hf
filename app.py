#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from model import Model

DESCRIPTION = '''# [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper)

- This demo only accepts as input `.obj` files with less than 100,000 faces.
- Inference takes about 10 minutes on a T4 GPU.
'''
if (SPACE_ID := os.getenv('SPACE_ID')) is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column():
            input_shape = gr.Model3D(label='Input 3D mesh')
            text = gr.Text(label='Text')
            seed = gr.Slider(label='Seed',
                             minimum=0,
                             maximum=100000,
                             value=3,
                             step=1)
            guidance_scale = gr.Slider(label='Guidance scale',
                                       minimum=0,
                                       maximum=50,
                                       value=7.5,
                                       step=0.1)
            run_button = gr.Button('Run')
        with gr.Column():
            progress_text = gr.Text(label='Progress')
            with gr.Tabs():
                with gr.TabItem(label='Images from each viewpoint'):
                    viewpoint_images = gr.Gallery(show_label=False).style(
                        columns=4, height='auto')
                with gr.TabItem(label='Result 3D model'):
                    result_3d_model = gr.Model3D(show_label=False)
                with gr.TabItem(label='Output mesh file'):
                    output_file = gr.File(show_label=False)
    with gr.Row():
        examples = [
            ['shapes/dragon1.obj', 'a photo of a dragon', 0, 7.5],
            ['shapes/dragon2.obj', 'a photo of a dragon', 0, 7.5],
            ['shapes/eagle.obj', 'a photo of an eagle', 0, 7.5],
            ['shapes/napoleon.obj', 'a photo of Napoleon Bonaparte', 3, 7.5],
            ['shapes/nascar.obj', 'A next gen nascar', 2, 10],
        ]
        gr.Examples(examples=examples,
                    inputs=[
                        input_shape,
                        text,
                        seed,
                        guidance_scale,
                    ],
                    outputs=[
                        result_3d_model,
                        output_file,
                    ],
                    cache_examples=False)

    run_button.click(fn=model.run,
                     inputs=[
                         input_shape,
                         text,
                         seed,
                         guidance_scale,
                     ],
                     outputs=[
                         viewpoint_images,
                         result_3d_model,
                         output_file,
                         progress_text,
                     ])

demo.queue(max_size=5).launch(debug=True)
