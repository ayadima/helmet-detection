export interface HelmetDetectionClass {
    name: string;
    id: number;
    displayName: string;
  }
  
  export const CLASSES: {[key: string]: HelmetDetectionClass} = {
    2: {
      name: 'person',
      id: 1,
      displayName: 'person',
    },
    1: {
      name: 'hat',
      id: 2,
      displayName: 'person with helmet',
    }
  };