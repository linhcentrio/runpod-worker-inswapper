import os
import io
import uuid
import base64
import copy
import cv2
import insightface
import numpy as np
import traceback
import runpod
import requests
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import List, Union
from PIL import Image
from minio import Minio
from urllib.parse import quote
from restoration import *
from schemas.input import INPUT_SCHEMA

FACE_SWAP_MODEL = 'checkpoints/inswapper_128.onnx'
TMP_PATH = '/tmp/inswapper'
logger = RunPodLogger()

# MinIO Configuration
MINIO_ENDPOINT = "108.181.198.160:9000"
MINIO_ACCESS_KEY = "a9TFRtBi8q3Nvj5P5Ris"
MINIO_SECRET_KEY = "fCFngM7YTr6jSkBKXZ9BkfDdXrStYXm43UGa0OZQ"
MINIO_BUCKET = "aiclipdfl"
MINIO_SECURE = False

# Initialize MinIO client
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(model_path: str,
                      torch_device: str,
                      det_size=(320, 320)):

    if torch_device == 'cuda':
        providers=['CUDAExecutionProvider']
    else:
        providers=['CPUExecutionProvider']

    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root="./checkpoints",
        providers=providers
    )

    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def swap_face(source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    global FACE_SWAPPER

    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)


def process(job_id: str,
            source_img: Union[Image.Image, List],
            target_img: Image.Image,
            source_indexes: str,
            target_indexes: str):

    global MODEL, FACE_ANALYSER

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(FACE_ANALYSER, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        if num_target_faces == 0:
            raise Exception('The target image does not contain any faces!')

        temp_frame = copy.deepcopy(target_img)

        if isinstance(source_img, list) and num_source_images == num_target_faces:
            logger.info('Replacing the faces in the target image from left to right by order', job_id)
            for i in range(num_target_faces):
                source_faces = get_many_faces(FACE_ANALYSER, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception('No source faces found!')

                temp_frame = swap_face(
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(FACE_ANALYSER, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            logger.info(f'Source faces: {num_source_faces}', job_id)
            logger.info(f'Target faces: {num_target_faces}', job_id)

            if source_faces is None or num_source_faces == 0:
                raise Exception('No source faces found!')

            if target_indexes == "-1":
                if num_source_faces == 1:
                    logger.info('Replacing the first face in the target image with the face from the source image', job_id)
                    num_iterations = num_source_faces
                elif num_source_faces < num_target_faces:
                    logger.info(f'There are less faces in the source image than the target image, replacing the first {num_source_faces} faces', job_id)
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    logger.info(f'There are less faces in the target image than the source image, replacing {num_target_faces} faces', job_id)
                    num_iterations = num_target_faces
                else:
                    logger.info('Replacing all faces in the target image with the faces from the source image', job_id)
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            elif source_indexes == '-1' and target_indexes == '-1':
                logger.info('Replacing specific face(s) in the target image with the face from the source image', job_id)
                target_indexes = target_indexes.split(',')
                source_index = 0

                for target_index in target_indexes:
                    target_index = int(target_index)

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                logger.info('Replacing specific face(s) in the target image with specific face(s) from the source image', job_id)

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception('Number of source indexes is greater than the number of faces in the source image')

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception('Number of target indexes is greater than the number of faces in the target image')

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f'Source index {source_index} is higher than the number of faces in the source image')

                        if target_index > num_target_faces-1:
                            raise ValueError(f'Target index {target_index} is higher than the number of faces in the target image')

                        temp_frame = swap_face(
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            logger.error('Unsupported face configuration', job_id)
            raise Exception('Unsupported face configuration')
        result = temp_frame
    else:
        logger.error('No target faces found', job_id)
        raise Exception('No target faces found!')

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def face_swap(job_id: str,
              src_img_path,
              target_img_path,
              source_indexes,
              target_indexes,
              background_enhance,
              face_restore,
              face_upsample,
              upscale,
              codeformer_fidelity,
              output_format):

    global TORCH_DEVICE, CODEFORMER_DEVICE, CODEFORMER_NET

    source_img_paths = src_img_path.split(';')
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    target_img = Image.open(target_img_path)

    try:
        logger.info('Performing face swap', job_id)
        result_image = process(
            job_id,
            source_img,
            target_img,
            source_indexes,
            target_indexes
        )
        logger.info('Face swap complete', job_id)
    except Exception as e:
        raise

    if face_restore:
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        logger.info('Performing face restoration using CodeFormer', job_id)

        try:
            result_image = face_restoration(
                result_image,
                background_enhance,
                face_upsample,
                upscale,
                codeformer_fidelity,
                upsampler,
                CODEFORMER_NET,
                CODEFORMER_DEVICE
            )
        except Exception as e:
            raise

        logger.info('CodeFormer face restoration completed successfully', job_id)
        result_image = Image.fromarray(result_image)

    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format=output_format)
    image_data = output_buffer.getvalue()

    return base64.b64encode(image_data).decode('utf-8')


def determine_file_extension(image_data):
    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.png'
    except Exception as e:
        image_extension = '.png'

    return image_extension


def download_file(url: str, local_path: str) -> bool:
    """Download file from URL with progress tracking"""
    try:
        logger.info(f'📥 Downloading {url}')
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        logger.info(f'✅ Downloaded: {local_path} ({downloaded/1024/1024:.1f} MB)')
        return True
        
    except Exception as e:
        logger.error(f'❌ Download failed: {e}')
        return False


def upload_to_minio(local_path: str, object_name: str) -> str:
    """Upload file to MinIO with enhanced error handling"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f'Local file not found: {local_path}')
        
        minio_client.fput_object(MINIO_BUCKET, object_name, local_path)
        file_url = f'http://{MINIO_ENDPOINT}/{MINIO_BUCKET}/{quote(object_name)}'
        logger.info(f'✅ Uploaded successfully: {file_url}')
        return file_url
        
    except Exception as e:
        logger.error(f'❌ Upload failed: {e}')
        raise e


def clean_up_temporary_files(source_image_path: str, target_image_path: str):
    os.remove(source_image_path)
    os.remove(target_image_path)


def face_swap_api(job_id: str, job_input: dict):
    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_image_data = job_input.get('source_image')
    target_image_data = job_input.get('target_image')
    
    # Check if input is URL or base64
    use_minio_output = job_input.get('use_minio_output', False)
    
    # Handle source image (URL or base64)
    if source_image_data.startswith('http'):
        # Download from URL
        source_image_path = f'{TMP_PATH}/source_{unique_id}.jpg'
        if not download_file(source_image_data, source_image_path):
            return {'error': 'Failed to download source image from URL'}
    else:
        # Decode base64
        source_image = base64.b64decode(source_image_data)
        source_file_extension = determine_file_extension(source_image_data)
        source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'
        
        # Save the source image to disk
        with open(source_image_path, 'wb') as source_file:
            source_file.write(source_image)

    # Handle target image (URL or base64)
    if target_image_data.startswith('http'):
        # Download from URL
        target_image_path = f'{TMP_PATH}/target_{unique_id}.jpg'
        if not download_file(target_image_data, target_image_path):
            clean_up_temporary_files(source_image_path, '')
            return {'error': 'Failed to download target image from URL'}
    else:
        # Decode base64
        target_image = base64.b64decode(target_image_data)
        target_file_extension = determine_file_extension(target_image_data)
        target_image_path = f'{TMP_PATH}/target_{unique_id}{target_file_extension}'
        
        # Save the target image to disk
        with open(target_image_path, 'wb') as target_file:
            target_file.write(target_image)

    try:
        logger.info(f'Source indexes: {job_input["source_indexes"]}', job_id)
        logger.info(f'Target indexes: {job_input["target_indexes"]}', job_id)
        logger.info(f'Background enhance: {job_input["background_enhance"]}', job_id)
        logger.info(f'Face Restoration: {job_input["face_restore"]}', job_id)
        logger.info(f'Face Upsampling: {job_input["face_upsample"]}', job_id)
        logger.info(f'Upscale: {job_input["upscale"]}', job_id)
        logger.info(f'Codeformer Fidelity: {job_input["codeformer_fidelity"]}', job_id)
        logger.info(f'Output Format: {job_input["output_format"]}', job_id)
        logger.info(f'Use MinIO Output: {use_minio_output}', job_id)

        result_image = face_swap(
            job_id,
            source_image_path,
            target_image_path,
            job_input['source_indexes'],
            job_input['target_indexes'],
            job_input['background_enhance'],
            job_input['face_restore'],
            job_input['face_upsample'],
            job_input['upscale'],
            job_input['codeformer_fidelity'],
            job_input['output_format']
        )

        clean_up_temporary_files(source_image_path, target_image_path)

        # Return result based on output preference
        if use_minio_output:
            # Save result to temporary file and upload to MinIO
            output_filename = f'inswapper_{job_id}_{uuid.uuid4().hex[:8]}.{job_input["output_format"].lower()}'
            temp_output_path = f'{TMP_PATH}/{output_filename}'
            
            # Decode base64 and save to file
            result_bytes = base64.b64decode(result_image)
            with open(temp_output_path, 'wb') as f:
                f.write(result_bytes)
            
            # Upload to MinIO
            try:
                output_url = upload_to_minio(temp_output_path, output_filename)
                os.remove(temp_output_path)  # Clean up temp file
                
                return {
                    'image_url': output_url,
                    'status': 'completed'
                }
            except Exception as upload_error:
                logger.error(f'MinIO upload failed: {upload_error}', job_id)
                os.remove(temp_output_path)  # Clean up temp file
                # Fallback to base64
                return {
                    'image': result_image,
                    'status': 'completed',
                    'minio_upload_failed': True
                }
        else:
            # Return base64 encoded image
            return {
                'image': result_image,
                'status': 'completed'
            }
            
    except Exception as e:
        logger.error(f'An exception was raised: {e}', job_id)
        clean_up_temporary_files(source_image_path, target_image_path)

        return {
            'error': str(e),
            'output': traceback.format_exc(),
            'refresh_worker': True
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    job_id = event['id']
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    return face_swap_api(job_id, validated_input['validated_input'])


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.join(script_dir, FACE_SWAP_MODEL)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), MODEL)
    logger.info(f'Face swap model: {MODEL}')

    if torch.cuda.is_available():
        TORCH_DEVICE = 'cuda'
    else:
        TORCH_DEVICE = 'cpu'

    logger.info(f'Torch device: {TORCH_DEVICE.upper()}')
    FACE_ANALYSER = get_face_analyser(MODEL, TORCH_DEVICE)
    FACE_SWAPPER = get_face_swap_model(model_path)

    # Ensure that CodeFormer weights have been successfully downloaded,
    # otherwise download them
    check_ckpts()

    logger.info('Setting upsampler to RealESRGAN_x2plus')
    upsampler = set_realesrgan()
    CODEFORMER_DEVICE = torch.device(TORCH_DEVICE)

    CODEFORMER_NET = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256'],
    ).to(CODEFORMER_DEVICE)

    ckpt_path = os.path.join(script_dir, 'CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth')
    logger.info(f'Loading CodeFormer model: {ckpt_path}')
    codeformer_checkpoint = torch.load(ckpt_path)['params_ema']
    CODEFORMER_NET.load_state_dict(codeformer_checkpoint)
    CODEFORMER_NET.eval()

    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )
