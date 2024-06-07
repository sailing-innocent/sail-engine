const path = require('path');
const fs = require('fs');
const { CleanWebpackPlugin } = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');

const appDirectory = fs.realpathSync(process.cwd());

module.exports = {
    entry: {
        main: path.resolve(appDirectory, './src/index.tsx'),
    },
    mode: "development",
    module: {
        rules: [
            {
                test: /\.(c|sc)ss$/,
                use: [
                    "style-loader",
                    "@teamsupercell/typings-for-css-modules-loader",
                    {
                      loader: "css-loader",
                      options: { modules: true }
                    },
                    "sass-loader"
                ]
            },
            {
                test: /\.tsx?$/,
                use: 'ts-loader',
                exclude: [
                    /node_modules/
                ]
            },
            {
                test: /\.(png|jpg|jpeg|svg|gif|pdf)$/,
                use: "file-loader"
            },
            {
                test: /\.(md|markdown)$/,
                use: "file-loader"
            },
            { 
                test: /\.wgsl$/, 
                type: "asset/source" 
            }
        ]
    },
    resolve: {
        extensions: ['.tsx', '.ts', '.js'],
        alias: {
            '@': path.resolve(appDirectory, "./src/")
        }
    },
    devServer: {
        host: "0.0.0.0",
        port: 8080, //port that we're using for local host (localhost:8080)
        static: path.resolve(appDirectory, "public"), //tells webpack to serve from the public folder
        hot: true,
        devMiddleware: {
            publicPath: "/",
        }
    },
    plugins: [
        new CleanWebpackPlugin(),
        new HtmlWebpackPlugin({
            inject: true,
            template: path.resolve(appDirectory, 'public/index.html'),
        }),
    ],
    output: {
        path: path.join(__dirname, 'dist'),
        filename: '[name].bundle.js'
    },
};
