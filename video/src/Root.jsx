import { Composition } from "remotion";
import { Demo, TOTAL } from "./Demo";

export const Root = () => (
  <Composition id="FootprintDemo" component={Demo} durationInFrames={TOTAL} fps={30} width={1920} height={1080} />
);
