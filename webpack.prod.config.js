const merge = require('webpack-merge')
const UglifyEsPlugin = require('uglify-es-webpack-plugin')
const common = require('./webpack.common.config.js')
const webpack = require('webpack')

module.exports = merge(common, {
  plugins: [
    new UglifyEsPlugin({
      compress:{
        drop_console: true
      }
    }),
    new webpack.DefinePlugin({
      'process.env.NODE_ENV': JSON.stringify('production')
    })
  ]
});