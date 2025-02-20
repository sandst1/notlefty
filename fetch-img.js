const getCarouselImageUrls = () => {
    // Find all slides within carousels
    const slides = document.querySelectorAll('.image-carousel__slide');
    
    // Extract img URLs from slides that contain images
    const urls = Array.from(slides)
      .map(slide => slide.querySelector('img'))
      .filter(img => img) // Remove null values (slides without images)
      .map(img => img.src);
      
    return urls;
  };