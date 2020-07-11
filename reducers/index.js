import { combineReducers } from "redux";
import globalReducer from "./globalReducer";

export default combineReducers({
  global: globalReducer,
});
