<div align="center">

# Inswapper Face Swapping | RunPod Serverless Worker

This is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker that uses roop ([inswapper](
https://huggingface.co/deepinsight/inswapper/tree/main)) for face
swapping AI tasks.

![Docker Pulls](https://img.shields.io/docker/pulls/ashleykza/runpod-worker-inswapper?style=for-the-badge&logo=docker&label=Docker%20Pulls&link=https%3A%2F%2Fhub.docker.com%2Frepository%2Fdocker%2Fashleykza%2Frunpod-worker-inswapper%2Fgeneral)
![Worker Version](https://img.shields.io/github/v/tag/ashleykleynhans/runpod-worker-inswapper?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDI2LjUuMywgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPgo8c3ZnIHZlcnNpb249IjEuMSIgaWQ9IkxheWVyXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IgoJIHZpZXdCb3g9IjAgMCAyMDAwIDIwMDAiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDIwMDAgMjAwMDsiIHhtbDpzcGFjZT0icHJlc2VydmUiPgo8c3R5bGUgdHlwZT0idGV4dC9jc3MiPgoJLnN0MHtmaWxsOiM2NzNBQjc7fQo8L3N0eWxlPgo8Zz4KCTxnPgoJCTxwYXRoIGNsYXNzPSJzdDAiIGQ9Ik0xMDE3Ljk1LDcxMS4wNGMtNC4yMiwyLjM2LTkuMTgsMy4wMS0xMy44NiwxLjgyTDM4Ni4xNyw1NTUuM2MtNDEuNzItMTAuNzYtODYuMDItMC42My0xMTYuNiwyOS43MwoJCQlsLTEuNCwxLjM5Yy0zNS45MiwzNS42NS0yNy41NSw5NS44LDE2Ljc0LDEyMC4zbDU4NC4zMiwzMjQuMjNjMzEuMzYsMTcuNCw1MC44Miw1MC40NSw1MC44Miw4Ni4zMnY4MDYuNzYKCQkJYzAsMzUuNDktMzguNDEsNTcuNjctNjkuMTUsMzkuOTRsLTcwMy4xNS00MDUuNjRjLTIzLjYtMTMuNjEtMzguMTMtMzguNzgtMzguMTMtNjYuMDJWNjY2LjYzYzAtODcuMjQsNDYuNDUtMTY3Ljg5LDEyMS45Mi0yMTEuNjYKCQkJTDkzMy44NSw0Mi4xNWMyMy40OC0xMy44LDUxLjQ3LTE3LjcsNzcuODMtMTAuODRsNzQ1LjcxLDE5NC4xYzMxLjUzLDguMjEsMzYuOTksNTAuNjUsOC41Niw2Ni41N0wxMDE3Ljk1LDcxMS4wNHoiLz4KCQk8cGF0aCBjbGFzcz0ic3QwIiBkPSJNMTUyNy43NSw1MzYuMzhsMTI4Ljg5LTc5LjYzbDE4OS45MiwxMDkuMTdjMjcuMjQsMTUuNjYsNDMuOTcsNDQuNzMsNDMuODIsNzYuMTVsLTQsODU3LjYKCQkJYy0wLjExLDI0LjM5LTEzLjE1LDQ2Ljg5LTM0LjI1LDU5LjExbC03MDEuNzUsNDA2LjYxYy0zMi4zLDE4LjcxLTcyLjc0LTQuNTktNzIuNzQtNDEuOTJ2LTc5Ny40MwoJCQljMC0zOC45OCwyMS4wNi03NC45MSw1NS4wNy05My45Nmw1OTAuMTctMzMwLjUzYzE4LjIzLTEwLjIxLDE4LjY1LTM2LjMsMC43NS00Ny4wOUwxNTI3Ljc1LDUzNi4zOHoiLz4KCQk8cGF0aCBjbGFzcz0ic3QwIiBkPSJNMTUyNC4wMSw2NjUuOTEiLz4KCTwvZz4KPC9nPgo8L3N2Zz4K&logoColor=%23ffffff&label=Worker%20Version&color=%23673ab7)
[![RunPod](https://api.runpod.io/badge/ashleykleynhans/runpod-worker-inswapper)](https://www.runpod.io/console/hub/ashleykleynhans/runpod-worker-inswapper)

</div>

## Model

The worker uses the `inswapper_128.onnx` model by [InsightFace](
https://insightface.ai/).

## Testing

1. [Local Testing](docs/testing/local.md)
2. [RunPod Testing](docs/testing/runpod.md)

## Building the Docker image that will be used by the Serverless Worker

[Building the Docker image](docs/building.md)

## RunPod API Endpoint

You can send requests to your RunPod API Endpoint using the `/run`
or `/runsync` endpoints.

Requests sent to the `/run` endpoint will be handled asynchronously,
and are non-blocking operations.  Your first response status will always
be `IN_QUEUE`.  You need to send subsequent requests to the `/status`
endpoint to get further status updates, and eventually the `COMPLETED`
status will be returned if your request is successful.

Requests sent to the `/runsync` endpoint will be handled synchronously
and are blocking operations.  If they are processed by a worker within
90 seconds, the result will be returned in the response, but if
the processing time exceeds 90 seconds, you will need to handle the
response and request status updates from the `/status` endpoint until
you receive the `COMPLETED` status which indicates that your request
was successful.

### RunPod API Examples

* [Swap as many source faces as possible into as many target faces as possible](
docs/api/swap-as-many-faces-as-possible.md)
* [Swap a single source face into a specific target face in a target image containing multiple faces](
docs/api/swap-single-source-face-into-specific-target-face.md)
* [Swap two faces from source image into 2 specific target faces in a target image containing multiple faces](
docs/api/swap-two-faces-into-specific-target-faces.md)
* [Swap two specific faces from source image containing multiple faces into 2 specific target faces in a target image containing multiple faces](
  docs/api/swap-specific-faces-into-specific-target-faces.md)

### Endpoint Status Codes

| Status      | Description                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------|
| IN_QUEUE    | Request is in the queue waiting to be picked up by a worker.  You can call the `/status` endpoint to check for status updates.  |
| IN_PROGRESS | Request is currently being processed by a worker.  You can call the `/status` endpoint to check for status updates.             |
| FAILED      | The request failed, most likely due to encountering an error.                                                                   |
| CANCELLED   | The request was cancelled.  This usually happens when you call the `/cancel` endpoint to cancel the request.                    |
| TIMED_OUT   | The request timed out.  This usually happens when your handler throws some kind of exception that does return a valid response. |
| COMPLETED   | The request completed successfully and the output is available in the `output` field of the response.                           |

## Serverless Handler

The serverless handler (`handler.py`) is a Python script that handles
the API requests to your Endpoint using the [runpod](https://github.com/runpod/runpod-python)
Python library.  It defines a function `handler(event)` that takes an
API request (event), runs the inference using the [inswapper](
https://huggingface.co/deepinsight/inswapper/tree/main) model (and
CodeFormer where applicable) with the `input`, and returns the `output`
in the JSON response.

## Acknowledgements

- [Inswapper](https://github.com/haofanwang/inswapper)
- [Roop](https://github.com/s0md3v/roop)
- [Insightface](https://github.com/deepinsight)
- [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer)
- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)
- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)

## Additional Resources

- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)
- [Getting Started With RunPod Serverless](https://trapdoor.cloud/getting-started-with-runpod-serverless/)
- [Serverless | Create a Custom Basic API](https://blog.runpod.io/serverless-create-a-basic-api/)

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/ashleykleynhans/runpod-worker-inswapper)
are welcome. Bug fixes and new features are encouraged.

## Appreciate my work?

<a href="https://www.buymeacoffee.com/ashleyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Features

- **Face Swapping**: Replace faces in target images with faces from source images
- **Multiple Face Support**: Handle multiple faces with specific indexing
- **Face Restoration**: Optional CodeFormer face enhancement
- **MinIO Integration**: Upload results to MinIO storage
- **URL Input Support**: Accept images from URLs or base64 encoded data
- **Flexible Output**: Return base64 encoded images or MinIO URLs

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `source_image` | string | Yes | - | Source image (base64 or URL) |
| `target_image` | string | Yes | - | Target image (base64 or URL) |
| `source_indexes` | string | No | "-1" | Source face indexes (comma-separated) |
| `target_indexes` | string | No | "-1" | Target face indexes (comma-separated) |
| `background_enhance` | boolean | No | true | Enhance background |
| `face_restore` | boolean | No | true | Apply face restoration |
| `face_upsample` | boolean | No | true | Upsample face |
| `upscale` | integer | No | 1 | Upscale factor |
| `codeformer_fidelity` | float | No | 0.5 | CodeFormer fidelity (0.0-1.0) |
| `output_format` | string | No | "JPEG" | Output format (JPEG/PNG) |
| `use_minio_output` | boolean | No | false | Upload result to MinIO |

## Usage Examples

### Basic Face Swap (Base64 Input/Output)
```json
{
  "source_image": "base64_encoded_source_image",
  "target_image": "base64_encoded_target_image",
  "source_indexes": "0",
  "target_indexes": "0",
  "face_restore": true,
  "output_format": "JPEG"
}
```

### Face Swap with MinIO Output
```json
{
  "source_image": "base64_encoded_source_image",
  "target_image": "base64_encoded_target_image",
  "source_indexes": "0",
  "target_indexes": "0",
  "face_restore": true,
  "output_format": "JPEG",
  "use_minio_output": true
}
```

### Face Swap with URL Input
```json
{
  "source_image": "https://example.com/source.jpg",
  "target_image": "https://example.com/target.jpg",
  "source_indexes": "0,1",
  "target_indexes": "0,1",
  "face_restore": true,
  "output_format": "PNG",
  "use_minio_output": true
}
```

## Response Format

### Base64 Output
```json
{
  "image": "base64_encoded_result_image",
  "status": "completed"
}
```

### MinIO Output
```json
{
  "image_url": "http://108.181.198.160:9000/aiclipdfl/inswapper_job_id_hash.jpg",
  "status": "completed"
}
```

### Error Response
```json
{
  "error": "Error message",
  "output": "Full error traceback",
  "refresh_worker": true
}
```

## Face Indexing

- Use `-1` to select all faces
- Use specific indexes (0, 1, 2, etc.) to select individual faces
- Faces are ordered from left to right in the image
- Multiple indexes can be specified as comma-separated values

## MinIO Configuration

The worker is configured to use the following MinIO settings:
- Endpoint: `108.181.198.160:9000`
- Bucket: `aiclipdfl`
- Access Key: `a9TFRtBi8q3Nvj5P5Ris`
- Secret Key: `fCFngM7YTr6jSkBKXZ9BkfDdXrStYXm43UGa0OZQ`

## Deployment

The worker is designed to run on RunPod serverless with CUDA support. The Dockerfile includes:

- CUDA 12.4.1 with cuDNN
- PyTorch 2.6.0 with CUDA support
- InsightFace InSwapper model
- CodeFormer face restoration
- MinIO client for storage integration

## Troubleshooting

### Common Issues

1. **Face Detection Failed**: Ensure images contain clear, visible faces
2. **Download Timeout**: Check network connectivity for URL inputs
3. **MinIO Upload Failed**: Verify MinIO server accessibility
4. **Memory Issues**: Reduce image resolution or disable face restoration

### Logs

The worker provides detailed logging for debugging:
- Face detection results
- Processing steps
- Download/upload status
- Error details

## License

This project is licensed under the MIT License.
