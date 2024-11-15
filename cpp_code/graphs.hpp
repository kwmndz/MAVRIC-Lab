#ifndef SVG_PLOTTER_HPP
#define SVG_PLOTTER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace plotting {

struct Point {
    double x;
    double y;
    
    // Constructor for easy initialization
    Point(double x_ = 0.0, double y_ = 0.0) : x(x_), y(y_) {}
};

class SVGPlotter {
private:
    static Point transformCoord(double x, double y) {
        const double SCALE = 9;  // Scale factor to make the plot larger
        const double OFFSET = 900;  // Center point offset
        return Point(x * SCALE , -y * SCALE + OFFSET);  // Flip y-axis for SVG coordinates
    }

    static std::string colorWithOpacity(const std::string& color, double opacity) {
        if (opacity == 1.0) return color;
        return color + std::to_string(static_cast<int>(opacity * 255));
    }

public:
    static void plotMovementInteractive2D(
        const std::vector<Point>& ugvPos,
        const Point& goalPos,
        const std::vector<Point>& obstacles,
        const std::string& logDir,
        const std::string& simId,
        const std::vector<double>& obsContainer,
        const Point& etLocalMin
    ) {
        std::ofstream svg_file(logDir + "/movement_" + simId + ".svg");
        if (!svg_file.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }

        // SVG header
        svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
                 << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"900\" viewBox=\"0 0 900 900\">\n"
                 << "<style>\n"
                 << "  .scanline { stroke: orange; stroke-width: 1; stroke-dasharray: 5,5; opacity: 0.5; }\n"
                 << "  .path { stroke: blue; stroke-width: 2; fill: none; }\n"
                 << "  .point { stroke-width: 1; }\n"
                 << "</style>\n";

        // Draw grid lines
        svg_file << "<g stroke=\"#eee\" stroke-width=\"1\">\n";
        for (int i = 0; i < 900; i += 50) {
            svg_file << "<line x1=\"" << i << "\" y1=\"0\" x2=\"" << i << "\" y2=\"900\"/>\n";
            svg_file << "<line x1=\"0\" y1=\"" << i << "\" x2=\"900\" y2=\"" << i << "\"/>\n";
        }
        svg_file << "</g>\n";

        // Draw UGV path
        svg_file << "<path d=\"M";
        for (size_t i = 0; i < ugvPos.size(); ++i) {
            Point p = transformCoord(ugvPos[i].x, ugvPos[i].y);
            svg_file << p.x << " " << p.y;
            if (i < ugvPos.size() - 1) svg_file << " L ";
        }
        svg_file << "\" class=\"path\"/>\n";

        // // Draw scanner traces
        // for (const auto& pos : ugvPos) {
        //     svg_file << "<path d=\"";
        //     const int numPoints = 50;
        //     for (int i = 0; i < numPoints; ++i) {
        //         double theta = -M_PI/2.0 + (i * M_PI)/(numPoints-1);
        //         double scan_x = pos.x + 5.0 * std::cos(theta);
        //         double scan_y = pos.y + 5.0 * std::sin(theta);
        //         Point p = transformCoord(scan_x, scan_y);
        //         svg_file << (i == 0 ? "M" : "L") << p.x << " " << p.y << " ";
        //     }
        //     svg_file << "\" class=\"scanline\"/>\n";
        // }

        // Draw obstacles
        for (const auto& obs : obstacles) {
            Point p = transformCoord(obs.x, obs.y);
            svg_file << "<circle cx=\"" << p.x << "\" cy=\"" << p.y 
                    << "\" r=\"3\" class=\"point\" fill=\"red\"/>\n";
        }

        // // Draw obstacle container
        // if (obsContainer.size() >= 4) {
        //     Point p1 = transformCoord(obsContainer[0], obsContainer[1]);
        //     Point p2 = transformCoord(obsContainer[2], obsContainer[3]);
            
        //     // Calculate container corners
        //     double dx = obsContainer[2] - obsContainer[0];
        //     double dy = obsContainer[3] - obsContainer[1];
        //     double length = std::sqrt(dx*dx + dy*dy);
        //     double nx = -dy/length * 6.25;  // Half width = 12.5/2
        //     double ny = dx/length * 6.25;
            
        //     std::vector<Point> corners = {
        //         transformCoord(obsContainer[0] + nx, obsContainer[1] + ny),
        //         transformCoord(obsContainer[0] - nx, obsContainer[1] - ny),
        //         transformCoord(obsContainer[2] - nx, obsContainer[3] - ny),
        //         transformCoord(obsContainer[2] + nx, obsContainer[3] + ny)
        //     };

        //     // Draw container area
        //     svg_file << "<path d=\"M";
        //     for (size_t i = 0; i < corners.size(); ++i) {
        //         svg_file << corners[i].x << " " << corners[i].y;
        //         if (i < corners.size() - 1) svg_file << " L ";
        //     }
        //     svg_file << " Z\" fill=\"rgba(255,0,0,0.2)\" stroke=\"purple\"/>\n";
        // }

        // Draw start, goal, and local minimum points
        Point start = transformCoord(ugvPos[0].x, ugvPos[0].y);
        Point goal = transformCoord(goalPos.x, goalPos.y);
        Point localMin = transformCoord(etLocalMin.x, etLocalMin.y);

        svg_file << "<circle cx=\"" << start.x << "\" cy=\"" << start.y 
                << "\" r=\"5\" class=\"point\" fill=\"green\"/>\n"
                << "<circle cx=\"" << goal.x << "\" cy=\"" << goal.y 
                << "\" r=\"5\" class=\"point\" fill=\"black\"/>\n"
                << "<circle cx=\"" << localMin.x << "\" cy=\"" << localMin.y 
                << "\" r=\"5\" class=\"point\" fill=\"black\"/>\n";

        // Add legend
        // svg_file << "<g transform=\"translate(650,50)\" font-family=\"Arial\" font-size=\"12\">\n"
        //         << "<text y=\"20\">Legend:</text>\n"
        //         << "<circle cx=\"10\" cy=\"40\" r=\"3\" fill=\"green\"/><text x=\"20\" y=\"45\">Start</text>\n"
        //         << "<circle cx=\"10\" cy=\"60\" r=\"3\" fill=\"black\"/><text x=\"20\" y=\"65\">Goal</text>\n"
        //         << "<circle cx=\"10\" cy=\"80\" r=\"3\" fill=\"red\"/><text x=\"20\" y=\"85\">Obstacles</text>\n"
        //         << "<line x1=\"10\" y1=\"100\" x2=\"30\" y2=\"100\" class=\"scanline\"/>"
        //         << "<text x=\"35\" y=\"105\">Scanner</text>\n"
        //         << "</g>\n";

        // Close SVG
        svg_file << "</svg>";
        svg_file.close();
    }
};

} // namespace plotting

#endif // SVG_PLOTTER_HPP