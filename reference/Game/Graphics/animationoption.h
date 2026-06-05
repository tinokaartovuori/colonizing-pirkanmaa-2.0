/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: animationoption.h, header for AnimationOption-class          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef ANIMATIONOPTION_H
#define ANIMATIONOPTION_H

#include <vector>
#include <string>

namespace Student {

/**
 * @brief The AnimationOption class defines options for animations
 *        every animated object has
 */
class AnimationOption
{
public:

    AnimationOption();

    /**
     * @brief AnimationOption constructor
     * @param onoff tells if the animation is wanted to be se on or off
     */
    AnimationOption(bool onoff);


    /**
     * @brief AnimationOption constructor
     * @param onoff tells if the animation is wanted to be se on or off
     * @param style is a string of the animation style. These are either
     *        "rollover" or "backandforth"
     */
    AnimationOption(bool onoff, std::string style);


    /**
     * @brief AnimationOption constructor
     * @param onoff tells if the animation is wanted to be se on or off
     * @param style is a string of the animation style. These are either
     *        "rollover" or "backandforth"
     * @param randomFrame
     */
    AnimationOption(bool onoff, std::string style, bool randomFrame);


    /**
     * @brief Returns either:
     * @return True: the item is set to be animated
     *         False: the item is se to not animated
     */
    bool isAnimated() const;


    /**
     * @brief Returns either:
     * @return True: the randomFrame option is enabled on the item
     *         False: the randomFrame option is disabled on the item
     */
    bool startRandomFrame() const;

    /**
     * @brief Returns:
     * @return string of the animation style. These are either
     *        "rollover" or "backandforth"
     */
    std::string getStyle() const;


private:
    bool m_animated; //Is the item set to be animated or not

    //String of the animation style. These are either "rollover" or "backandforth"
    std::string m_style;

    //Is the randomFrame option enabled or not
    bool m_randomFrame;

};
}
#endif // ANIMATIONOPTION_H
