const path = require("path");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require("mini-css-extract-plugin");
const CopyPlugin = require("copy-webpack-plugin");

module.exports = {
  entry: ["./index.js"],
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "js/app.bundle.js",
    publicPath: "/",
  },
  module: {
    rules: [
      {
        test: /\.(sa|sc|c)ss$/,
        use: [
          {
            loader: MiniCssExtractPlugin.loader,
            options: {
              hmr: process.env.NODE_ENV === "development",
            },
          },
          "css-loader",
          "sass-loader",
        ],
      },
      {
        test: /\.(jpe?g|png|gif|svg|ico)$/,
        use: [
          {
            loader: "url-loader",
            options: {
              limit: 40000,
              name: "static/[name].[contenthash].[ext]",
            },
          },
          "image-webpack-loader",
        ],
      },
      {
        test: /\.woff(2)?(\?v=[0-9]\.[0-9]\.[0-9])?$/,
        use: "url-loader",
      },
      {
        test: /\.(ttf|eot|svg)(\?v=[0-9]\.[0-9]\.[0-9])?$/,
        use: "url-loader",
      },
      {
        test: /\.(js|jsx)$/,
        exclude: /(node_modules|bower_components)/,
        use: "babel-loader",
      },
    ],
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: "Kapil Sharma",
      minify: {
        collapseWhitespace: true,
      },
      template: "./index.html",
      favicon: "./static/favicon.ico",
    }),
    new MiniCssExtractPlugin({
      test: /\.css$/,
      use: [
        {
          loader: MiniCssExtractPlugin.loader,
        },
        "css-loader",
      ],
    }),
    new CopyPlugin({
      patterns: [{ from: "./CNAME" }],
    }),
  ],
  resolve: {
    extensions: [".js", ".jsx"],
  },
};
