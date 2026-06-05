/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: sceneitem.h, header for SceneItem-class                      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef SceneItem_H
#define SceneItem_H

#include <QGraphicsItem>
#include <QPainter>

#include <memory>
#include <map>

#include "Core/gameobject.h"
#include "DAL/objectmanager.h"
#include "Graphics/animationoption.h"
#include "Overlays/mousehoverborder.h"

namespace Student {

/**
 * @brief The SceneItem class is a custom QGraphicsItem that
 *        has additional features such as animation frames
 */
class SceneItem : public QGraphicsItem
{
public:
    /**
     * @brief Constructor
     * @param obj shared_ptr to the obj.
     */
    SceneItem(const std::shared_ptr<Course::BaseObject> &obj);

    /**
     * @brief getBoundObject
     * @return the object this item is bound to.
     */
    const std::shared_ptr<Course::BaseObject> &getBoundObject();


    /**
     * @brief Checks if the given GameObject is represented with this
     *        SceneItem
     * @return True: it is
     *         False: it isn't
     */
    bool isSameObj(std::shared_ptr<Course::BaseObject> obj);


    /**
     * @brief Returns the rectangle that is surrounding the item
     * @return QRectF
     */
    virtual QRectF boundingRect() const override;


    /**
     * @brief Sets a pixmap vector for the item that can be accessed
     *        later to draw the pixmaps easily
     */
    void setItemPixmap();


    /**
     * @brief Sets the animation frame manually. This doesn't help much
     *        of the item is animated.
     * @param Integer of the frame that is wanted to be set
     * @post Exception guarantee: No-throw
     */
    void setAnimationFrame(int frame);


    /**
     * @brief Sets the animation frame manually. This doesn't help much
     *        of the item is animated.
     * @param Integer of the frame that is wanted to be set
     */
    void setRandomImageIndex();


    /**
     * @brief Changes the animation frame according to the item's animation
     *        options. This method is called from the GameScene method that
     *        is connected to the timer
     */
    void changeAnimationFrame();


    /**
     * @brief Sets the grid size for the sceneitem. Grid size tells how many
     *        pixels wide and tall one grid(tile) is
     * @param gridSize is an integer of the grid size
     */
    void setGridSize(int gridSize);


    /**
     * @brief Sets the width of the sceneitem in pixels
     * @param integer of the width in pixels
     */
    void setWidth(int width);


    /**
     * @brief Sets the height of the sceneitem in pixels
     * @param integer of the height in pixels
     */
    void setHeight(int height);


    /**
     * @brief Returns the type of the SceneItem. If the item is not derived
     *        from this class the type is "SceneItem
     * @return string of the type
     */
    virtual std::string getType();


    /**
     * @brief Changes the animation option of the SceneItem
     * @param The wanted AnimationOption class object to be set
     */
    void setAnimationOption(AnimationOption ani);

protected:
    const std::shared_ptr<Course::BaseObject> baseObject_;

    int currentImageFrame_; //Current frame to be drawn

    int animationDirection_; //Direction of the next frame. One means it
                             //will go forwards and -1 backwards

    AnimationOption animationOption_; //Will the image frame roll over
                                               //or back and forth

    std::vector<QPixmap> itemPixmap_; //Vector of the item's frames in QPixmap type

    bool randomizeStart_ = false; //Is the animation randomized
    int width_; //Width in pixels
    int height_; //Height in pixels

    int gridSize_; //How many pixels one tile is wide
};

}
#endif // SceneItem_H
