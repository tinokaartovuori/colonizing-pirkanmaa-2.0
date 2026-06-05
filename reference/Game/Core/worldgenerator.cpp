/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: worldgenerator.cpp, see worldgenerator.h for more info       #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#include "Core/worldgenerator.h"

#include "Tiles/forest.h"
#include "Tiles/grassland.h"
#include "Tiles/mountain.h"
#include "Buildings/mikontalo.h"
#include "Tiles/river.h"
#include "Tiles/abundantforest.h"

#include "Graphics/imagevectors.h"

#include <QDebug>
#include <vector>
#include <iostream>
#include <math.h>
#include <tuple>


namespace Student {
WorldGenerator& WorldGenerator::getInstance()
{
    static WorldGenerator instance;
    return instance;
}

void WorldGenerator::generateMap(unsigned int size_x_,
        unsigned int size_y_,
        unsigned int seed,
        const std::weak_ptr<Course::iObjectManager> &objectmanager,
        const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
        const std::weak_ptr<GameSettingsManager> &gamesettingsmanager,
        std::weak_ptr<GameScene> scene)
{
    this->addConstructor<Course::Forest>("Forest");
    this->addConstructor<Course::Grassland>("Grassland");
    this->addConstructor<Student::Mountain>("Mountain");
    this->addConstructor<Student::AbundantForest>("AbundantForest");
    this->addConstructor<Student::River>("River");

    srand(seed);

    std::vector<std::shared_ptr<Course::TileBase>> tiles;

    std::vector<std::vector<int>> matrix = generateTerrain(size_x_, size_y_);

    for (unsigned int x = 0; x < size_x_; ++x)
    {
        for (unsigned int y = 0; y < size_y_; ++y)
        {
            TileConstructorPointer ctor;
            std::vector<std::string> imageVector;
            Student::AnimationOption animationOption;

            bool spawnMikontalo = false;

            if (matrix.at(y).at(x) == 0) {
                ctor = m_ctors.find("Grassland")->second;
                imageVector = ImageVectors::GRASSLAND;
                animationOption = AnimationOptions::GRASSLAND;
            }
            else if (matrix.at(y).at(x) == 1) {
                ctor = m_ctors.find("Forest")->second;
                int rnd = (rand() % 2) - 1;
                if (rnd == 0) {
                    imageVector = ImageVectors::FOREST_1;
                }
                else {
                    imageVector = ImageVectors::FOREST_2;
                }
                animationOption = AnimationOptions::FOREST;
            }
            else if (matrix.at(y).at(x) == 2) {
                ctor = m_ctors.find("Mountain")->second;
                imageVector = ImageVectors::MOUNTAIN;
                animationOption = AnimationOptions::MOUNTAIN;
            }
            else if (matrix.at(y).at(x) == 4) {
                ctor = m_ctors.find("River")->second;
                imageVector = std::get<0>(getRiverOrientation(x, y, matrix, 4));

                animationOption = AnimationOptions::RIVER;
            }
            else if (matrix.at(y).at(x) == 5) {
                ctor = m_ctors.find("Mountain")->second;
                imageVector = ImageVectors::MOUNTAIN_FOREST;
                animationOption = AnimationOptions::MOUNTAIN_FOREST;
            }
            else if (matrix.at(y).at(x) == 3) {
                // Spawn Mikontalo to tile too.
                ctor = m_ctors.find("Grassland")->second;
                imageVector = ImageVectors::GRASSLAND;
                animationOption = AnimationOptions::GRASSLAND;
                spawnMikontalo = true;
            }
            else if (matrix.at(y).at(x) == 6) {
                ctor = m_ctors.find("AbundantForest")->second;
                imageVector = ImageVectors::ABUNDANT_FOREST;
                animationOption = AnimationOptions::FOREST;
            }
            else {
                ctor = m_ctors.find("Grassland")->second;
                imageVector = ImageVectors::GRASSLAND;
                animationOption = AnimationOptions::GRASSLAND;
            }

            tiles.push_back(ctor(Course::Coordinate(x, y), 1, 1,
                                 eventhandler.lock(), objectmanager.lock()));

            if (spawnMikontalo) {

                std::shared_ptr<Student::Mikontalo> mikontalo = std::make_shared<Student::Mikontalo>(
                            eventhandler.lock(),
                            objectmanager.lock(),
                            nullptr
                            );

                mikontalo->setImageFiles(ImageVectors::MIKONTALO);
                tiles.back()->addBuilding(mikontalo);
                eventhandler.lock()->updateTile(tiles.back());

            }

            if (std::dynamic_pointer_cast<Student::River>(tiles.back()) != nullptr) {
                std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverOrientation(std::get<1>(getRiverOrientation(x, y, matrix, 4)));
                if (imageVector == ImageVectors::RIVER_EW) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("EW");
                }
                if (imageVector == ImageVectors::RIVER_NS) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("NS");
                }
                if (imageVector == ImageVectors::RIVER_NE) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("NE");
                }
                if (imageVector == ImageVectors::RIVER_NW) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("NW");
                }
                if (imageVector == ImageVectors::RIVER_SW) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("SW");
                }
                if (imageVector == ImageVectors::RIVER_SE) {
                    std::dynamic_pointer_cast<Student::River>(tiles.back())->setRiverShape("SE");
                }
            }

            tiles.back()->setGameSettings(gamesettingsmanager.lock());
            tiles.back()->setImageFiles(imageVector);
            tiles.back()->setAnimationOption(animationOption);
            scene.lock()->drawItem(tiles.back());
        }
    }

    objectmanager.lock()->addTiles(tiles);

}

std::vector<std::vector<int>> WorldGenerator::generateTerrain(int size_x_, int size_y_) {

    std::vector<std::vector<int>> matrix;
    std::vector<int> row;

    int rnd;
    for (int y = 0; y < size_y_; ++y)
    {
        for (int x = 0; x < size_x_; ++x)
        {
            rnd = rand() % 100 + 1;

            if (rnd < 15) {
                row.push_back(1);
            }
            else {
                row.push_back(0);
            }
        }
        matrix.push_back(row);
        row.clear();
    }
    std::vector<std::vector<int>> temp_matrix;
    temp_matrix = matrix;
    for (int y = 0; y < size_y_; ++y)
    {
        for (int x = 0; x < size_x_; ++x)
        {
            if (temp_matrix.at(y).at(x) == 1) {

                try {
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y-1).at(x-1) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y-1).at(x) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y-1).at(x+1) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y).at(x-1) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y).at(x+1) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y+1).at(x-1) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y+1).at(x) = 1;
                    }
                    if (rand() % 100 + 1 > 40) {
                        matrix.at(y+1).at(x+1) = 1;
                    }
                }
                catch(...) {
                    continue;
                }

            }
        }
    }

    int dir = rand() % 2 - 1;
    int starting_tile_x = (rand() % (size_x_ - 4)) + 2;
    int starting_tile_y = (rand() % (size_y_ - 4)) + 2;
    int current_x = 0;
    int current_y = 0;

    int next_dir;

    std::vector<int> last_dirs = {0, 0};

    if (dir == 0) {
        current_x = starting_tile_x;
    } else {
        current_y = starting_tile_y;
    }

    while (true) {

        if (current_x >= size_x_ or current_x < 0) break;
        if (current_y >= size_y_ or current_y < 0) break;

        matrix.at(current_y).at(current_x) = 4;


        if (last_dirs.back() == 0 && last_dirs.at(last_dirs.size() - 2) == 0 ) {
            next_dir = rand() % 3;
        }
        else if (last_dirs.back() == 0 && last_dirs.at(last_dirs.size() - 2) == 1) {
            next_dir = (rand() % 2 - 1);
        }
        else if (last_dirs.back() == 0 && last_dirs.at(last_dirs.size() - 2) == 2) {
            next_dir = (rand() % 2 - 1) * 2;
        }
        else if (last_dirs.back() == 1) {
            next_dir = (rand() % 2 - 1);
        }
        else if (last_dirs.back() == 2) {
            next_dir = (rand() % 2 - 1) * 2;
        }
        else {
            next_dir = 0;
        }

        if (dir == 0) {
            if (next_dir == 0) {
                current_y += 1;
            }
            if (next_dir == 1) {
                current_x += 1;
            }
            if (next_dir == 2) {
                current_x -= 1;
            }
        } else {
            if (next_dir == 0) {
                current_x += 1;
            }
            if (next_dir == 1) {
                current_y += 1;
            }
            if (next_dir == 2) {
                current_y -= 1;
            }
        }

        last_dirs.push_back(next_dir);


    }

    for (int i = 0; i < (int)(rand() % (int)round(size_x_ * size_y_ * 0.30)) + 4; i++) {
        int rnd_x = rand() % size_x_;
        int rnd_y = rand() % size_y_;


        if (matrix.at(rnd_y).at(rnd_x) == 1) {
            matrix.at(rnd_y).at(rnd_x) = 5;
        }
        if (matrix.at(rnd_y).at(rnd_x) == 0) {
            matrix.at(rnd_y).at(rnd_x) = 2;
        }
        else {
            continue;
        }
    }

    for (int x = 0; x < (int)(size_x_*size_y_)/30; x++) {
        int rnd_x = rand() % size_x_;
        int rnd_y = rand() % size_y_ ;

        if (matrix.at(rnd_y).at(rnd_x) == 4) continue;

        matrix.at(rnd_y).at(rnd_x) = 6;
    }

    while (true) {
        int rnd_x = rand() % size_x_;
        int rnd_y = rand() % size_y_ ;

        if (matrix.at(rnd_y).at(rnd_x) == 4) continue;

        matrix.at(rnd_y).at(rnd_x) = 3;
        break;
    }

    return matrix;
}


std::tuple<std::vector<std::string>, int> WorldGenerator::getRiverOrientation
                     (int x_, int y_, std::vector<std::vector<int>> matrix_, int num)
{
    std::vector<std::vector<int>> matrix = matrix_;
    int x = x_;
    int y = y_;
    int size_x_ = matrix.at(0).size();
    int size_y_ = matrix.size();

    try {
        if (matrix.at(y-1).at(x) == num and matrix.at(y).at(x-1) == num) {
            return std::make_tuple(ImageVectors::RIVER_NW, 3);
        }
        if (matrix.at(y-1).at(x) == num and matrix.at(y).at(x+1) == num) {
            return std::make_tuple(ImageVectors::RIVER_NE, 3);
        }
        if (matrix.at(y+1).at(x) == num and matrix.at(y).at(x-1) == num) {
            return std::make_tuple(ImageVectors::RIVER_SW, 3);
        }
        if (matrix.at(y+1).at(x) == num and matrix.at(y).at(x+1) == num) {
            return std::make_tuple(ImageVectors::RIVER_SE, 3);
        }
        if (matrix.at(y-1).at(x) == num and matrix.at(y+1).at(x) == num) {
            return std::make_tuple(ImageVectors::RIVER_NS, 1);
        }
        if (matrix.at(y).at(x-1) == num and matrix.at(y).at(x+1) == num) {
            return std::make_tuple(ImageVectors::RIVER_EW, 0);
        }
    }
    catch (...) {
        try {
            if (x == 0 and y == 0) {
                if (matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                if (matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
            else if (x == 0 and y == size_y_ - 1) {
                if (matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                if (matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
            else if (x == size_x_ -1 and y == 0) {
                if (matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                if (matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
            else if (x == size_x_ - 1 and y == size_y_ - 1) {
                if (matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                if (matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
            else if (y == 0) {
                if (matrix.at(y).at(x-1) == num and matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
                else if (matrix.at(y+1).at(x) == num and matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SW, 3);
                }
                else if (matrix.at(y+1).at(x) == num and matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SE, 3);
                }
                else if (matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NW, 3);
                }
                else if (matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NE, 3);
                }
                else {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
            }
            else if (y == size_y_ - 1) {
                if (matrix.at(y).at(x-1) == num and matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
                else if (matrix.at(y-1).at(x) == num and matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NW, 3);
                }
                else if (matrix.at(y-1).at(x) == num and matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NE, 3);
                }
                else if (matrix.at(y).at(x-1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SW, 3);
                }
                else if (matrix.at(y).at(x+1) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SE, 3);
                }
                else {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
            }
            else if (x == 0) {
                if (matrix.at(y-1).at(x) == num and matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                else if (matrix.at(y).at(x+1) == num and matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NE, 3);
                }
                else if (matrix.at(y).at(x+1) == num and matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SE, 3);
                }
                else if (matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NW, 3);
                }
                else if (matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SW, 3);
                }
                else {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
            else if (x == size_x_ - 1) {
                if (matrix.at(y-1).at(x) == num and matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NS, 1);
                }
                else if (matrix.at(y).at(x-1) == num and matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NW, 3);
                }
                else if (matrix.at(y).at(x-1) == num and matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SW, 3);
                }
                else if (matrix.at(y-1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_NE, 3);
                }
                else if (matrix.at(y+1).at(x) == num) {
                    return std::make_tuple(ImageVectors::RIVER_SE, 3);
                }
                else {
                    return std::make_tuple(ImageVectors::RIVER_EW, 0);
                }
            }
        }
        catch (...) {
            qDebug() << "Error with river orientation";
        }
    }

    return {};
}

} // namespace Course
