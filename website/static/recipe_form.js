document.addEventListener('DOMContentLoaded', function() {
  const imageUpload = document.getElementById('imageUpload');
  const videoUpload = document.getElementById('videoUpload');
  const imageInput = imageUpload.querySelector('input[type="file"]');
  const videoInput = videoUpload.querySelector('input[type="file"]');
  const imagePreviewContainer = document.getElementById('imagePreviewContainer');
  const videoPreviewContainer = document.getElementById('videoPreviewContainer');

  const form = document.getElementById('wf-form-Recipe-Form');
  const removedFiles = new Set();

  function createPreviewItem(file, isExisting = false) {
    const previewItem = document.createElement('div');
    previewItem.className = 'preview-item';

    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.src = isExisting ? file.src : URL.createObjectURL(file);
      img.alt = 'Recipe Image';
      img.className = 'thumbnail';
      previewItem.appendChild(img);
    } else if (file.type.startsWith('video/')) {
      const video = document.createElement('video');
      video.src = isExisting ? file.src : URL.createObjectURL(file);
      video.controls = true;
      video.className = 'thumbnail';
      previewItem.appendChild(video);
    }

    const removeBtn = document.createElement('button');
    removeBtn.innerHTML = '&times';
    removeBtn.className = 'remove-btn';
    removeBtn.type = 'button';
    removeBtn.dataset.filename = isExisting ? file.filename : file.name;
    removeBtn.addEventListener('click', function(e) {
      e.stopPropagation();  // Prevent event from bubbling up
      previewItem.remove();
      if (isExisting) {
        removedFiles.add(file.filename);
      }
    });
    previewItem.appendChild(removeBtn);

    return previewItem;
  } //end function create Preview item

  function handleFiles(files, previewContainer, isImage) {
    if (isImage) {
      for (const file of files) {
        const previewItem = createPreviewItem(file);
        const mediaElement = previewItem.querySelector('img');
        if (mediaElement) {
          mediaElement.file = file;
        }
        previewContainer.appendChild(previewItem);
        console.log(`Added image preview:`, file.name);
      }
    } else {
      // For video, replace existing preview (if any) with the new one
      previewContainer.innerHTML = '';
      if (files.length > 0) {
        const file = files[0];
        const previewItem = createPreviewItem(file);
        const mediaElement = previewItem.querySelector('video');
        if (mediaElement) {
          mediaElement.file = file;
        }
        previewContainer.appendChild(previewItem);
        console.log(`Added video preview:`, file.name);
      }
    }
  }

  function triggerUpload(uploadBox, input) {
    if (event.target === uploadBox || 
        event.target.tagName === 'I' || 
        event.target.tagName === 'P' ||
        event.target === imagePreviewContainer ||
        event.target === videoPreviewContainer) {
      input.click();
    }
  }

  imageUpload.addEventListener('click', (event) => triggerUpload(imageUpload, imageInput));
  videoUpload.addEventListener('click', (event) => triggerUpload(videoUpload, videoInput));

  imageInput.addEventListener('change', (e) => handleFiles(e.target.files, imagePreviewContainer, true));
  videoInput.addEventListener('change', (e) => handleFiles(e.target.files, videoPreviewContainer, false));

  // Handle drag and drop
  [imageUpload, videoUpload].forEach(uploadBox => {
    uploadBox.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadBox.classList.add('dragover');
    });

    uploadBox.addEventListener('dragleave', () => {
      uploadBox.classList.remove('dragover');
    });

    uploadBox.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadBox.classList.remove('dragover');
      const files = e.dataTransfer.files;
      const isImage = uploadBox === imageUpload;
      handleFiles(files, isImage ? imagePreviewContainer : videoPreviewContainer, isImage);
    });
  });

  // Handle existing images and video
  const existingImages = imagePreviewContainer.querySelectorAll('.preview-item');
  existingImages.forEach(item => {
    const removeBtn = item.querySelector('.remove-btn');
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      item.remove();
      removedFiles.add(removeBtn.dataset.filename);
    });
  });

  const existingVideo = videoPreviewContainer.querySelector('.preview-item');
  if (existingVideo) {
    const removeBtn = existingVideo.querySelector('.remove-btn');
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      existingVideo.remove();
      removedFiles.add(removeBtn.dataset.filename);
    });
  }

  // Handle form submission
  form.addEventListener('submit', function(e) {
    e.preventDefault();
    const instructionsList = document.getElementById('instructionsList');
    const hiddenInput = document.getElementById('Instructions');

    function addNewStep() {
      const li = document.createElement('li');
      const textarea = document.createElement('textarea');
      textarea.className = 'instruction-input w-input';
      textarea.placeholder = 'Enter a step and press Enter for a new line';
      textarea.rows = 1;
      li.appendChild(textarea);
      instructionsList.appendChild(li);
      textarea.focus();
      autoResize(textarea);
    }

    function updateHiddenInput() {
      const steps = Array.from(instructionsList.querySelectorAll('textarea'))
        .map(textarea => textarea.value.trim())
        .filter(step => step !== '');
      hiddenInput.value = steps.join('|');
    }

    function autoResize(textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = textarea.scrollHeight + 'px';
    }

    instructionsList.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        addNewStep();
      }
    });

    instructionsList.addEventListener('input', function(e) {
      if (e.target.tagName === 'TEXTAREA') {
        autoResize(e.target);
        updateHiddenInput();
      }
    });

    // Auto-resize existing textareas
    instructionsList.querySelectorAll('textarea').forEach(autoResize);

    // Update hidden input before form submission
    document.querySelector('form').addEventListener('submit', updateHiddenInput);
    const formData = new FormData(form);

    // Validate and combine tags and ingredients
    const tagsInput = window.addedTags;

    const ingredientsInput = window.addedIngredients;
    
    const allTags = new Set([...tagsInput]);
    const allIngredients = new Set([...ingredientsInput]);

    formData.set('Tags', Array.from(allTags).join(','));
    formData.set('Ingredients', Array.from(allIngredients).join(','));

    console.log("Submitting form...");

    // Add existing images
    const existingImages = imagePreviewContainer.querySelectorAll('.preview-item img[data-filename]');
    console.log("Existing images:", existingImages.length);
    existingImages.forEach((img, index) => {
      formData.append(`existing_images_${index}`, img.dataset.filename);
    });

    // Add new images
    const newImages = imagePreviewContainer.querySelectorAll('.preview-item img:not([data-filename])');
    console.log("New images:", newImages.length);
    newImages.forEach((img, index) => {
      const file = img.file || img.src;
      console.log(`Processing new image ${index}:`, file);
      if (file instanceof File) {
        formData.append(`new_images_${index}`, file, file.name);
        console.log(`Appended new image ${index} as File`);
      } else if (typeof file === 'string' && file.startsWith('blob:')) {
        fetch(file)
          .then(res => res.blob())
          .then(blob => {
            formData.append(`new_images_${index}`, blob, `new_image_${index}.png`);
            console.log(`Appended new image ${index} as Blob`);
          });
      }
    });

    // Handle video (existing or new)
    const videoPreview = videoPreviewContainer.querySelector('.preview-item video');
    if (videoPreview) {
      if (videoPreview.dataset.filename) {
        formData.append('existing_video', videoPreview.dataset.filename);
      } else if (videoPreview.file) {
        formData.append('new_video', videoPreview.file, videoPreview.file.name);
      }
    }

    // Send the form data to the server
    Promise.all(Array.from(formData.values()).filter(value => value instanceof Promise))
      .then(() => {
      fetch(form.action, {
        method: 'POST',
        body: formData,
        redirect: 'follow' // This tells fetch to follow redirects
      }).then(response => {
        if (response.ok) {
          // Check if the response is a redirect
          if (response.redirected) {
            window.location.href = response.url; // Go to the URL the server redirected to
          } else {
            // Handle non-redirect successful response
            console.log('Form submitted successfully');
            // Optionally, you can redirect to a default page if no redirect was received
            // window.location.href = '/profile';
          }
        } else {
          // Handle error responses
          console.error('Form submission failed');
        }
      }).catch(error => {
        console.error('Error:', error);
      });
    });
  });
});