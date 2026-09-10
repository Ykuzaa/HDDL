def draw_localization(img, lab, title, image_size=IMAGE_SIZE):
    """Display one image and, if it holds an object, its bounding box in the class color."""
    colors = ["royalblue", "limegreen", "purple", "darkorange"] # Different colors for different classes
    classes = ["Buffalo", "Elephant", "Rhino", "Zebra"]

    # Image display
    plt.imshow(img)

    # Nothing else to draw when the label says that no object is present.
    # Every image does contain one in this part; this will change in Part II.
    if lab[0] <= 0.5:
        plt.title(title.format("no object"))
        return

    # Determining the class
    class_id = np.argmax(lab[5:])

    # Determining the coordinates of the bounding box in the image frame
    ax = (lab[1]*y_std[1] + y_mean[1]) * image_size
    ay = (lab[2]*y_std[2] + y_mean[2]) * image_size
    width = (lab[3]*y_std[3] + y_mean[3]) * image_size
    height = (lab[4]*y_std[4] + y_mean[4]) * image_size
    # Determining the extrema of the bounding box, namely the minimum and maximum x/y value
    p_x = [ax-width/2, ax+width/2]
    p_y = [ay-height/2, ay+height/2]
    # Display the bounding box in the right color
    plt.plot([p_x[0], p_x[0]], p_y, color=colors[class_id], linewidth=2)
    plt.plot([p_x[1], p_x[1]], p_y, color=colors[class_id], linewidth=2)
    plt.plot(p_x, [p_y[0], p_y[0]], color=colors[class_id], linewidth=2)
    plt.plot(p_x, [p_y[1], p_y[1]], color=colors[class_id], linewidth=2)

    plt.title(title.format(classes[class_id]))